from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from event_transformer import FIELDS, NUM_EVENT_FEATURES, NUM_SUMMARY_FEATURES, build_sequences, events_in_window, \
    load_data, make_vocab
from metric import precision_at_recall


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SequenceDataset(Dataset):
    def __init__(self, categorical, numerical, summaries, target=None):
        self.categorical, self.numerical, self.summaries = categorical, numerical, summaries
        self.target = None if target is None else np.asarray(target, dtype=np.float32)

    def __len__(self): return len(self.numerical)

    def __getitem__(self, i):
        cats = {f: torch.from_numpy(values[i]) for f, values in self.categorical.items()}
        result = (cats, torch.from_numpy(self.numerical[i]), torch.from_numpy(self.summaries[i]))
        return result if self.target is None else (*result, torch.tensor(self.target[i]))


class CausalGRU(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], hidden: int = 64):
        super().__init__()
        emb = 8
        self.embeddings = nn.ModuleDict(
            {f: nn.Embedding(size + 1, emb, padding_idx=0) for f, size in vocab_sizes.items()})
        self.input = nn.Sequential(nn.Linear(len(FIELDS) * emb + NUM_EVENT_FEATURES, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.next_event = nn.Linear(hidden, vocab_sizes['event_name'] + 1)
        self.head = nn.Sequential(nn.Linear(hidden + NUM_SUMMARY_FEATURES, 64), nn.Tanh(), nn.Dropout(.25),
                                  nn.Linear(64, 1))

    def forward(self, cats, nums, summaries):
        x = torch.cat([*(self.embeddings[f](cats[f]) for f in FIELDS), nums], dim=-1)
        states, _ = self.gru(self.input(x))
        lengths = cats['event_name'].ne(0).sum(dim=1).clamp_min(1)
        final = states[torch.arange(len(states), device=states.device), lengths - 1]
        return self.head(torch.cat([final, summaries], dim=1)).squeeze(-1), self.next_event(states), lengths


@torch.no_grad()
def predict(model, loader, device):
    model.eval();
    out = []
    for batch in loader:
        cats, nums, summaries = batch[:3]
        cats = {f: v.to(device) for f, v in cats.items()}
        logits, _, _ = model(cats, nums.to(device), summaries.to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--max-len', type=int, default=192)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    seed_everything();
    root = args.root.resolve()
    train, test, events = load_data(root)
    ev_train, ev_test = events_in_window(events, train), events_in_window(events, test)
    vocab = make_vocab(pd.concat([ev_train, ev_test], ignore_index=True))
    cat_tr, num_tr, sum_tr, _ = build_sequences(ev_train, train, vocab, args.max_len)
    cat_te, num_te, sum_te, _ = build_sequences(ev_test, test, vocab, args.max_len)
    valid = train.window_start_ts.ge('2026-04-17').to_numpy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CausalGRU({f: len(v) for f, v in vocab.items()}).to(device)
    print(f'device={device} parameters={sum(p.numel() for p in model.parameters()):,}')
    train_ds = SequenceDataset({f: v[~valid] for f, v in cat_tr.items()}, num_tr[~valid], sum_tr[~valid],
                               train.target.to_numpy()[~valid])
    val_ds = SequenceDataset({f: v[valid] for f, v in cat_tr.items()}, num_tr[valid], sum_tr[valid],
                             train.target.to_numpy()[valid])
    test_ds = SequenceDataset(cat_te, num_te, sum_te)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)
    pos = train.target.to_numpy()[~valid].sum();
    neg = (~valid).sum() - pos
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(3., neg / pos), device=device))
    next_loss = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_state, stale, history = -1., None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train();
        losses = []
        for cats, nums, summaries, target in train_loader:
            cats = {f: v.to(device) for f, v in cats.items()}
            logits, next_logits, lengths = model(cats, nums.to(device), summaries.to(device))
            next_targets = cats['event_name'][:, 1:]
            next_logits = next_logits[:, :-1]
            loss = cls_loss(logits, target.to(device)) + .10 * next_loss(next_logits.reshape(-1, next_logits.shape[-1]),
                                                                         next_targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True);
            loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(), 1.);
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = predict(model, val_loader, device);
        score = precision_at_recall(train.target.to_numpy()[valid], val_pred)
        history.append({'epoch': epoch, 'train_loss': np.mean(losses), 'val_p_at_r70': score})
        print(f'epoch={epoch:02d} loss={np.mean(losses):.4f} val_P@R70={score:.4f}', flush=True)
        if score > best:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 8: break
    model.load_state_dict(best_state)
    out = root / 'artifacts';
    out.mkdir(exist_ok=True)
    val_pred, test_pred = predict(model, val_loader, device), predict(model, test_loader, device)
    pd.DataFrame(history).to_csv(out / 'causal_gru_history.csv', index=False)
    pd.DataFrame({'cookie_id': train.loc[valid, 'cookie_id'], 'target': train.target.to_numpy()[valid],
                  'causal_gru_score': val_pred}).to_csv(out / 'causal_gru_val_predictions.csv', index=False)
    pd.DataFrame({'cookie_id': test.cookie_id, 'causal_gru_score': test_pred}).to_csv(
        out / 'causal_gru_test_predictions.csv', index=False)
    print(f'best validation P@R70={best:.4f}')
    base_path = out / 'validation_predictions.csv'
    if base_path.exists():
        d = pd.read_csv(base_path).merge(pd.read_csv(out / 'causal_gru_val_predictions.csv'),
                                         on=['cookie_id', 'target'])
        rows = [(w, precision_at_recall(d.target, (1 - w) * d.ensemble_score + w * d.causal_gru_score)) for w in
                np.linspace(0, 1, 21)]
        print('best diversity blend:', max(rows, key=lambda x: x[1]))


if __name__ == '__main__': main()
