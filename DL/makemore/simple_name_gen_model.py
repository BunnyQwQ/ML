import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import nll_loss
from pathlib import Path


class NameGenModel:
    def __init__(self, file_path: str, seed: int = 42):
        self.g = torch.Generator().manual_seed(seed)

        self.df = self._load_and_clean_data(file_path)

        self.chars = sorted(list(set(''.join(self.df))))
        self.stoi = {'.': 0, **{s: i + 1 for i, s in enumerate(self.chars)}}
        self.itos = {b: a for a, b in self.stoi.items()}

        self.N = self._count_bigrams()

        self.P = self._compute_probabilities()

    def _load_and_clean_data(self, file_path: str) -> pd.Series:
        df = pd.read_json(file_path, orient='records', lines=True)['text']
        df = df[~df.str.contains('.', regex=False)]
        df = df.str.lower()
        df = df.str.replace('p', 'р').str.replace('h', 'н')
        return df

    def _count_bigrams(self) -> torch.Tensor:
        n_tokens = len(self.stoi)
        N = torch.zeros((n_tokens, n_tokens), dtype=torch.int32)
        for word in self.df:
            chars = ['.'] + list(word) + ['.']
            for ch1, ch2 in zip(chars, chars[1:]):
                N[self.stoi[ch1], self.stoi[ch2]] += 1
        return N

    def _compute_probabilities(self) -> torch.Tensor:
        P = (self.N + 1).float()
        return P / P.sum(dim=1, keepdim=True)

    def show(self):
        plt.imshow(self.N)
        plt.show()

    def generate(self, num: int) -> list[str]:
        results = []
        for _ in range(num):
            out, ix = [], 0
            while True:
                ix = torch.multinomial(
                    self.P[ix], num_samples=1, generator=self.g
                ).item()
                if ix == 0:
                    break
                out.append(self.itos[ix])
            results.append(''.join(out))
        return results

    def nll_loss(self, words=None):
        words = words if words is not None else self.df
        log = 0.0
        n = 0
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                i1, i2 = self.stoi[ch1], self.stoi[ch2]
                prob = self.P[i1, i2]
                log += torch.log(prob)
                n += 1
        nll = -log / n
        return nll.item()


s = NameGenModel(Path(__file__).resolve().parent.parent / "data" / "names_table.jsonl", seed=30)
s.show()
print(s.generate(10))
print(s.nll_loss())