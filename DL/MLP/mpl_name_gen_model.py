import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt


class NamesDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]


class MLPNameGenModel(nn.Module):
    def __init__(
        self,
        file_path: str,
        block_size: int = 3,
        embed_dim: int = 10,
        batch_size: int = 32,
        hidden_dim: int = 100,
        seed: int = 42,
        val_frac: float = 0.1,
        test_frac: float = 0.1
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.g = torch.Generator().manual_seed(seed)

        self.block_size = block_size
        self.embed_dim = embed_dim
        self.batch_size = batch_size

        self.df = self._load_and_clean_data(file_path)
        self.chars = sorted(set(''.join(self.df)))
        self.stoi = {'.': 0, **{s: i + 1 for i, s in enumerate(self.chars)}}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        X, Y = self._build_dataset(self.df)

        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = \
            self._split_data(X, Y, val_frac, test_frac)

        self.train_loader = DataLoader(
            NamesDataset(self.X_train, self.Y_train),
            batch_size=self.batch_size, shuffle=True, generator=self.g
        )

        self.C = nn.Parameter(torch.randn(self.vocab_size, embed_dim))
        self.linear1 = nn.Linear(block_size * embed_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.vocab_size)

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _load_and_clean_data(self, file_path: str) -> pd.Series:
        df = pd.read_json(file_path, orient='records', lines=True)['text']
        df = df[~df.str.contains('.', regex=False)]
        df = df.str.lower()
        df = df.str.replace('p', 'р').str.replace('h', 'н')
        return df

    def _build_dataset(self, words) -> tuple[torch.Tensor, torch.Tensor]:
        X, Y = [], []
        for word in words:
            context = [0] * self.block_size
            for ch in word + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        return torch.tensor(X), torch.tensor(Y)

    def _split_data(self, X, Y, val_frac, test_frac):
        if abs(val_frac) + abs(test_frac) > 1:
            raise ValueError("Невозможно поделить данные")
        val_frac = max(0, val_frac)
        test_frac = max(0, test_frac)
        n = X.shape[0]
        perm = torch.randperm(n, generator=self.g)
        X, Y = X[perm], Y[perm]

        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        n_train = n - n_val - n_test

        X_train, Y_train = X[:n_train], Y[:n_train]
        X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
        X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]
        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.C[x]                          # (batch, block_size, embed_dim)
        emb = emb.view(emb.shape[0], -1)          # (batch, block_size*embed_dim)
        pre = self.bn1(self.linear1(emb))         # linear -> BatchNorm (нормализуем ДО активации)
        h = torch.tanh(pre)                       # (batch, hidden_dim)
        logits = self.linear2(h)                  # (batch, vocab_size)
        return logits

    def train_model(self, epochs: int = 100, lr: float = 0.001, verbose: bool = True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in self.train_loader:
                logits = self.forward(xb)
                loss = F.cross_entropy(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / n_batches
            val_loss = self.evaluate(self.X_val, self.Y_val)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose:
                print(f"epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        return self.train_losses, self.val_losses

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        self.eval()
        logits = self.forward(X)
        loss = F.cross_entropy(logits, Y)
        return loss.item()

    def test_loss(self) -> float:
        return self.evaluate(self.X_test, self.Y_test)

    @torch.no_grad()
    def generate(self, num: int = 10) -> list[str]:
        self.eval()
        results = []
        for _ in range(num):
            out = []
            context = [0] * self.block_size
            while True:
                x = torch.tensor([context])
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs[0], num_samples=1, generator=self.g).item()
                if ix == 0:
                    break
                out.append(self.itos[ix])
                context = context[1:] + [ix]
            results.append(''.join(out))
        return results

    def plot_losses(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.train_losses, label='train')
        plt.plot(self.val_losses, label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Training and validation loss')
        plt.show()


m = MLPNameGenModel(file_path="names_table.jsonl")
m.train_model()
m.plot_losses()
print(m.test_loss())
print(m.generate())