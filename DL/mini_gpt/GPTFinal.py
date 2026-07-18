import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class Head(nn.Module):
    def __init__(self, C: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(C, head_size, bias=False)
        self.key = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape

        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)         # (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, -float('inf'))  # (B, T, T)
        weights = F.softmax(scores, dim=-1)                                 # (B, T, T)
        weights = self.dropout(weights)
        out = weights @ v                                                   # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, C: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert C % num_heads == 0, f"C={C} не делится на num_heads={num_heads}"
        head_size = C // num_heads
        self.heads = nn.ModuleList([
            Head(C, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.project = nn.Linear(C, C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.project(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, C: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 4 * C),
            nn.ReLU(),
            nn.Linear(4 * C, C),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, C: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(C, num_heads, block_size, dropout)
        self.ff = FeedForward(C, dropout)
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, C: int, num_heads: int, num_blocks: int,
                 block_size: int, dropout: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, C)
        self.position_embeddings = nn.Embedding(block_size, C)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(C, num_heads, block_size, dropout) for _ in range(num_blocks)
        ])
        self.lnf = nn.LayerNorm(C)
        self.lmf = nn.Linear(C, vocab_size)
        self.block_size = block_size

        self.lmf.weight = self.token_embeddings.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        tok_emb = self.token_embeddings(idx) # (B, T, C)
        pos_emb = self.position_embeddings(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lmf(x) # (B, T, vocab_size)
        return logits


class NamesDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]


class GPTFinal(nn.Module):
    def __init__(
        self,
        file_path: str,
        block_size: int = 16,
        embed_dim: int = 64,
        batch_size: int = 32,
        seed: int = 666,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        num_heads: int = 4,
        num_blocks: int = 3,
        dropout: float = 0.1
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

        self.gpt = GPT(
            vocab_size=self.vocab_size, C=self.embed_dim, num_heads=num_heads,
            num_blocks=num_blocks, block_size=self.block_size, dropout=dropout
        )

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
        bs = self.block_size
        for word in words:
            toks = [0] + [self.stoi[ch] for ch in word] + [0]
            toks = toks[:bs + 1]
            x = toks[:-1]
            y = toks[1:]
            pad = bs - len(x)
            x = x + [0] * pad
            y = y + [-1] * pad
            X.append(x)
            Y.append(y)
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
        return self.gpt(x)

    def train_model(self, epochs: int = 200, lr: float = 3e-4, weight_decay: float = 0.01, patience: int = 5, verbose: bool = True):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in self.train_loader:
                logits = self.forward(xb) # (B, T, vocab)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B * T, V), yb.view(B * T), ignore_index=-1)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / n_batches
            val_loss = self.evaluate(self.X_val, self.Y_val)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose:
                print(f"epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"early stopping на эпохе {epoch} (лучший val={best_val:.4f})")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return self.train_losses, self.val_losses

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        self.eval()
        logits = self.forward(X)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), Y.view(B * T), ignore_index=-1)
        return loss.item()

    def test_loss(self) -> float:
        return self.evaluate(self.X_test, self.Y_test)

    @torch.no_grad()
    def generate(self, num: int = 10, max_len: int = 30) -> list[str]:
        self.eval()
        results = []
        for _ in range(num):
            context = [0]
            out = []
            for _ in range(max_len):
                idx = torch.tensor([context[-self.block_size:]])
                logits = self.forward(idx)
                logits_last = logits[0, -1, :]
                probs = F.softmax(logits_last, dim=0)
                ix = torch.multinomial(probs, num_samples=1, generator=self.g).item()
                if ix == 0:
                    break
                out.append(self.itos[ix])
                context.append(ix)
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


if __name__ == '__main__':
    m = GPTFinal(file_path="names_table.jsonl")
    m.train_model()
    m.plot_losses()
    print("test loss:", m.test_loss())
    print(m.generate())