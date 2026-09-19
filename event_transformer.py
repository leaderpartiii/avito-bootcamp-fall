from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_mpl_cache = Path(__file__).parent / "artifacts" / "mplconfig"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset

from metric import precision_at_recall

SEED = 42
FIELDS = ["event_name", "platform", "item_category", "item_location", "seller_type"]
NUM_EVENT_FEATURES = 8
NUM_SUMMARY_FEATURES = 8


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(root: Path):
    train = pd.read_csv(root / "data/train.csv", parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"])
    test = pd.read_csv(root / "data/test.csv", parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"])
    events = pd.read_csv(root / "data/events.csv.gz", parse_dates=["event_ts"])
    return train, test, events


def events_in_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    x = events.merge(meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id", how="inner")
    return x.loc[(x.event_ts >= x.window_start_ts) & (x.event_ts < x.window_end_ts)].copy()


def make_vocab(events: pd.DataFrame) -> dict[str, dict[str, int]]:
    vocab = {}
    for col in FIELDS:
        values = events[col].fillna("__MISSING__").astype(str)
        vocab[col] = {value: i + 1 for i, value in enumerate(sorted(values.unique()))}
    return vocab


def build_sequences(events: pd.DataFrame, meta: pd.DataFrame, vocab: dict[str, dict[str, int]], max_len: int):
    x = events.sort_values(["cookie_id", "event_ts"]).copy()
    x["gap_sec"] = x.groupby("cookie_id").event_ts.diff().dt.total_seconds().fillna(0).clip(0, 86400)
    x["hour"] = x.event_ts.dt.hour
    x["hour_sin"] = np.sin(2 * np.pi * x.hour / 24.0)
    x["hour_cos"] = np.cos(2 * np.pi * x.hour / 24.0)
    x["gap_log"] = np.log1p(x.gap_sec) / np.log1p(86400)
    x["search_page_num"] = x.search_page.fillna(0).clip(0, 100) / 100.0
    x["has_item"] = x.item_id.notna().astype(float)
    x["has_query"] = x.search_query.notna().astype(float)
    x["has_pointer"] = x.pointer_x.notna().astype(float)
    x["elapsed_log"] = (
        x.groupby("cookie_id").event_ts.transform(lambda s: (s - s.min()).dt.total_seconds())
    )
    x["elapsed_log"] = np.log1p(x.elapsed_log.clip(0, 86400)) / np.log1p(86400)

    by_cookie = {cookie: part for cookie, part in x.groupby("cookie_id", sort=False)}
    n = len(meta)
    categorical = {field: np.zeros((n, max_len), dtype=np.int64) for field in FIELDS}
    numerical = np.zeros((n, max_len, NUM_EVENT_FEATURES), dtype=np.float32)
    summaries = np.zeros((n, NUM_SUMMARY_FEATURES), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)

    for row, cookie in enumerate(meta.cookie_id):
        part = by_cookie.get(cookie)
        if part is None:
            continue
        full_length = len(part)
        gaps = part.gap_sec.iloc[1:].to_numpy(dtype=np.float32)
        duration = max(0.0, (part.event_ts.iloc[-1] - part.event_ts.iloc[0]).total_seconds())
        summaries[row] = np.array([
            np.log1p(full_length) / np.log1p(512),
            np.log1p(min(duration, 86400)) / np.log1p(86400),
            np.log1p(np.mean(gaps) if len(gaps) else 0) / np.log1p(86400),
            np.log1p(np.median(gaps) if len(gaps) else 0) / np.log1p(86400),
            np.log1p(np.std(gaps) if len(gaps) else 0) / np.log1p(86400),
            np.mean(gaps <= 1) if len(gaps) else 0,
            np.mean(gaps <= 10) if len(gaps) else 0,
            part.event_name.nunique(dropna=False) / max(full_length, 1),
        ], dtype=np.float32)
        if full_length > max_len:
            part = part.iloc[np.linspace(0, full_length - 1, max_len, dtype=int)]
        length = len(part)
        lengths[row] = length
        for field in FIELDS:
            values = part[field].fillna("__MISSING__").astype(str)
            categorical[field][row, :length] = values.map(vocab[field]).fillna(0).to_numpy()
        numerical[row, :length, 0] = part.gap_log.to_numpy()
        numerical[row, :length, 1] = part.hour_sin.to_numpy()
        numerical[row, :length, 2] = part.hour_cos.to_numpy()
        numerical[row, :length, 3] = part.search_page_num.to_numpy()
        numerical[row, :length, 4] = part.has_item.to_numpy()
        numerical[row, :length, 5] = part.has_query.to_numpy()
        numerical[row, :length, 6] = part.has_pointer.to_numpy()
        numerical[row, :length, 7] = part.elapsed_log.to_numpy()
    return categorical, numerical, summaries, lengths


class CookieSequenceDataset(Dataset):
    def __init__(self, categorical, numerical, summaries, target=None):
        self.categorical = categorical
        self.numerical = numerical
        self.summaries = summaries
        self.target = None if target is None else np.asarray(target, dtype=np.float32)

    def __len__(self):
        return len(self.numerical)

    def __getitem__(self, index):
        cats = {field: torch.from_numpy(values[index]) for field, values in self.categorical.items()}
        nums = torch.from_numpy(self.numerical[index])
        summary = torch.from_numpy(self.summaries[index])
        if self.target is None:
            return cats, nums, summary
        return cats, nums, summary, torch.tensor(self.target[index])


class EventTransformer(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], max_len: int, d_model=96, nhead=4, layers=2, dropout=0.25,
                 emb_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {field: nn.Embedding(size + 1, emb_dim, padding_idx=0) for field, size in vocab_sizes.items()})
        self.input = nn.Sequential(nn.Linear(len(FIELDS) * emb_dim + NUM_EVENT_FEATURES, d_model),
                                   nn.LayerNorm(d_model), nn.GELU())
        self.position = nn.Embedding(max_len + 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + NUM_SUMMARY_FEATURES),
            nn.Linear(d_model + NUM_SUMMARY_FEATURES, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, cats, nums, summaries):
        pieces = [self.embeddings[field](cats[field]) for field in FIELDS]
        x = self.input(torch.cat(pieces + [nums], dim=-1))
        valid = cats["event_name"].ne(0)
        positions = torch.arange(1, x.shape[1] + 1, device=x.device).unsqueeze(0)
        x = x + self.position(positions)
        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.position.weight[0].view(1, 1, -1)
        x = torch.cat([cls, x], dim=1)
        valid = torch.cat([torch.ones((valid.shape[0], 1), dtype=torch.bool, device=valid.device), valid], dim=1)
        encoded = self.encoder(x, src_key_padding_mask=~valid)
        pooled = torch.cat([encoded[:, 0], summaries], dim=1)
        return self.head(pooled).squeeze(-1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions = []
    for batch in loader:
        cats, nums, summaries = batch[:3]
        cats = {field: value.to(device) for field, value in cats.items()}
        predictions.append(torch.sigmoid(model(cats, nums.to(device), summaries.to(device))).cpu().numpy())
    return np.concatenate(predictions)


def train_model(model, train_loader, val_loader, y_val, device, epochs=25, lr=2e-3, max_pos_weight=5.0):
    pos = float(train_loader.dataset.target.sum())
    neg = float(len(train_loader.dataset) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(max_pos_weight, neg / max(pos, 1.0))], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_score, best_state, wait = -1.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for cats, nums, summaries, target in train_loader:
            cats = {field: value.to(device) for field, value in cats.items()}
            logits = model(cats, nums.to(device), summaries.to(device))
            loss = criterion(logits, target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = predict(model, val_loader, device)
        score = precision_at_recall(y_val, val_pred)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_p_at_r70": float(score)})
        print(f"epoch={epoch:02d} train_loss={np.mean(losses):.4f} val_P@R70={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-pos-weight", type=float, default=5.0)
    args = parser.parse_args()
    seed_everything()
    root = args.root.resolve()
    train, test, events = load_data(root)
    ev_train, ev_test = events_in_window(events, train), events_in_window(events, test)
    vocab = make_vocab(pd.concat([ev_train, ev_test], ignore_index=True))
    cat_train, num_train, summary_train, _ = build_sequences(ev_train, train, vocab, args.max_len)
    cat_test, num_test, summary_test, _ = build_sequences(ev_test, test, vocab, args.max_len)
    is_valid = train.window_start_ts.ge("2026-04-17").to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventTransformer(
        {field: len(values) for field, values in vocab.items()},
        max_len=args.max_len,
        d_model=args.d_model,
        layers=args.layers,
        dropout=args.dropout,
        emb_dim=args.embedding_dim,
    ).to(device)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    train_ds = CookieSequenceDataset({f: v[~is_valid] for f, v in cat_train.items()}, num_train[~is_valid],
                                     summary_train[~is_valid], train.target.to_numpy()[~is_valid])
    val_ds = CookieSequenceDataset({f: v[is_valid] for f, v in cat_train.items()}, num_train[is_valid],
                                   summary_train[is_valid], train.target.to_numpy()[is_valid])
    test_ds = CookieSequenceDataset(cat_test, num_test, summary_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)
    score, history = train_model(
        model, train_loader, val_loader, train.target.to_numpy()[is_valid], device,
        args.epochs, args.learning_rate, args.max_pos_weight,
    )
    val_pred = predict(model, val_loader, device)
    test_pred = predict(model, test_loader, device)
    out = root / "artifacts"
    out.mkdir(exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(out / "transformer_training_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o")
    axes[0].set_title("Transformer train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(history_df["epoch"], history_df["val_p_at_r70"], marker="o", color="tab:orange")
    best_epoch = int(history_df.loc[history_df["val_p_at_r70"].idxmax(), "epoch"])
    best_value = float(history_df["val_p_at_r70"].max())
    axes[1].axvline(best_epoch, linestyle="--", color="gray", alpha=0.7)
    axes[1].set_title(f"Validation P@R≥0.70 (best={best_value:.4f})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("P@R≥0.70")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "transformer_training_curves.png", dpi=150)
    plt.close(fig)
    pd.DataFrame({"cookie_id": train.loc[is_valid, "cookie_id"], "target": train.loc[is_valid, "target"],
                  "transformer_score": val_pred}).to_csv(out / "transformer_val_predictions.csv", index=False)
    pd.DataFrame({"cookie_id": test.cookie_id, "transformer_score": test_pred}).to_csv(
        out / "transformer_test_predictions.csv", index=False)
    torch.save(model.state_dict(), out / "event_transformer.pt")
    print(f"best validation P@R70={score:.4f}")

    baseline_path = out / "validation_predictions.csv"
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        merged = base.merge(pd.read_csv(out / "transformer_val_predictions.csv"), on=["cookie_id", "target"])
        rows = []
        for weight in np.linspace(0, 1, 11):
            raw = (1 - weight) * merged["ensemble_score"] + weight * merged["transformer_score"]
            rows.append((weight, precision_at_recall(merged.target, raw)))
        best_weight, best_blend = max(rows, key=lambda item: item[1])
        print(f"best diversity blend transformer_weight={best_weight:.1f} P@R70={best_blend:.4f}")


if __name__ == "__main__":
    main()
