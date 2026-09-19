from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from metric import precision_at_recall


SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_features_from_notebook(root: Path):
    notebook = json.loads((root / "quickstart.ipynb").read_text())
    namespace: dict[str, object] = {}
    for cell_index in (1, 3, 7, 8, 9):
        exec("".join(notebook["cells"][cell_index]["source"]), namespace)
    return (
        namespace["train"],
        namespace["test"],
        namespace["Xtr"],
        namespace["Xte"],
        namespace["ytr"],
        namespace["feature_cols"],
    )


class TanhMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 512), nn.Tanh(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.Tanh(), nn.Dropout(0.20),
            nn.Linear(256, 64), nn.Tanh(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    for (features,) in loader:
        scores.append(torch.sigmoid(model(features.to(device))).cpu().numpy())
    return np.concatenate(scores)


def fit_and_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = TanhMLP(x_train.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= 300_000, f"MLP is too large: {n_params:,} parameters"
    print(f"device={device} parameters={n_params:,}")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_valid)), batch_size=batch_size * 2)
    pos_weight = min(3.0, float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-3)
    best_state, best_score, stale = None, -1.0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(device)), target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_pred = predict(model, valid_loader, device)
        score = precision_at_recall(y_valid, valid_pred)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_p_at_r70": score})
        print(f"epoch={epoch:02d} train_loss={np.mean(losses):.4f} val_P@R70={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 10:
                break
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    seed_everything()
    root = args.root.resolve()
    train, test, x_train_df, x_test_df, y, features = load_features_from_notebook(root)
    valid = train.window_start_ts.ge("2026-04-17").to_numpy()

    scaler = QuantileTransformer(
        n_quantiles=min(1000, int((~valid).sum())),
        output_distribution="normal",
        random_state=SEED,
    )
    x_fit = scaler.fit_transform(x_train_df.loc[~valid, features]).astype(np.float32)
    x_valid = scaler.transform(x_train_df.loc[valid, features]).astype(np.float32)
    x_test = scaler.transform(x_test_df[features]).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = fit_and_score(x_fit, y[~valid], x_valid, y[valid], device, args.epochs, args.batch_size)

    valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_valid)), batch_size=args.batch_size * 2)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=args.batch_size * 2)
    valid_pred = predict(model, valid_loader, device)
    test_pred = predict(model, test_loader, device)
    out = root / "artifacts"
    out.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(out / "mlp_training_history.csv", index=False)
    pd.DataFrame({"cookie_id": train.loc[valid, "cookie_id"], "target": y[valid], "mlp_score": valid_pred}).to_csv(out / "mlp_val_predictions.csv", index=False)
    pd.DataFrame({"cookie_id": test.cookie_id, "mlp_score": test_pred}).to_csv(out / "mlp_test_predictions.csv", index=False)
    torch.save({"model": model.state_dict(), "feature_cols": features}, out / "tabular_mlp.pt")
    print(f"best validation P@R70={max(x['val_p_at_r70'] for x in history):.4f}")

    baseline_path = out / "validation_predictions.csv"
    if baseline_path.exists():
        merged = pd.read_csv(baseline_path).merge(pd.read_csv(out / "mlp_val_predictions.csv"), on=["cookie_id", "target"])
        rows = []
        for weight in np.linspace(0, 1, 21):
            score = (1 - weight) * merged.ensemble_score + weight * merged.mlp_score
            rows.append((weight, precision_at_recall(merged.target, score)))
        weight, score = max(rows, key=lambda row: row[1])
        print(f"best diversity blend mlp_weight={weight:.2f} P@R70={score:.4f}")


if __name__ == "__main__":
    main()
