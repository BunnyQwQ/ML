import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class BigramNNModel(nn.Module):
    def __init__(self, file_path: str, seed: int = 42):
        super().__init__()
        self.g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)

        self.df = self._load_and_clean_data(file_path)
        self.chars = sorted(set(''.join(self.df)))
        self.stoi = {'.': 0, **{s: i + 1 for i, s in enumerate(self.chars)}}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        self.xs, self.ys = self._build_dataset()
        self.xs_onehot = F.one_hot(self.xs, num_classes=self.vocab_size).float()

        self.linear = nn.Linear(self.vocab_size, self.vocab_size, bias=False)

        self.losses: list[float] = []

    def _load_and_clean_data(self, file_path: str) -> pd.Series:
        df = pd.read_json(file_path, orient='records', lines=True)['text']
        df = df[~df.str.contains('.', regex=False)]
        df = df.str.lower()
        df = df.str.replace('p', 'р').str.replace('h', 'н')
        return df

    def _build_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for word in self.df:
            chars = ['.'] + list(word) + ['.']
            for ch1, ch2 in zip(chars, chars[1:]):
                xs.append(self.stoi[ch1])
                ys.append(self.stoi[ch2])
        return torch.tensor(xs), torch.tensor(ys)

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        return self.linear(x_onehot)

    def train_model(self, epochs: int = 100, lr: float = 1., reg: float = 0.01) -> list[float]:
        self.losses = []
        for epoch in range(epochs):
            logits = self.forward(self.xs_onehot)
            loss = F.cross_entropy(logits, self.ys) + reg * self.linear.weight.pow(2).mean()

            self.zero_grad()
            loss.backward()

            with torch.no_grad():
                for p in self.parameters():
                    p -= lr * p.grad

            self.losses.append(loss.item())
            if epoch % 20 == 0:
                print(f"epoch {epoch}: loss = {loss.item():.4f}")

        return self.losses

    @torch.no_grad()
    def nll_loss(self) -> float:
        logits = self.forward(self.xs_onehot)
        loss = F.cross_entropy(logits, self.ys)
        return loss.item()

    @torch.no_grad()
    def generate(self, num: int = 10) -> list[str]:
        results = []
        for _ in range(num):
            out, ix = [], 0
            while True:
                x_onehot = F.one_hot(torch.tensor([ix]), num_classes=self.vocab_size).float()
                logits = self.forward(x_onehot)
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs[0], num_samples=1, generator=self.g).item()
                if ix == 0:
                    break
                out.append(self.itos[ix])
            results.append(''.join(out))
        return results

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    def show_weights(self):
        with torch.no_grad():
            plt.imshow(self.linear.weight.detach())
            plt.colorbar()
            plt.show()


net = BigramNNModel(Path(__file__).resolve().parent.parent / "data" / "names_table.jsonl", seed=30)
net.train_model(epochs=500, lr=10.0)

print(net.generate(10))
print("Final NN loss:", net.nll_loss())

net.plot_loss()
net.show_weights()