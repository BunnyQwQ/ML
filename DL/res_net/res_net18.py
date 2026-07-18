import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # residual
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_ch, out_ch, s))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


class ResNetFinal(nn.Module):
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        seed: int = 42,
        val_frac: float = 0.1,
        num_workers: int = 2,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.g = torch.Generator().manual_seed(seed)

        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.train_loader, self.val_loader, self.test_loader = \
            self._build_loaders(data_dir, val_frac, num_workers)

        self.model = ResNet18(num_classes=10).to(self.device)

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accs: list[float] = []

    def _build_loaders(self, data_dir: str, val_frac: float, num_workers: int):
        train_aug = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=self.train_transform)
        train_plain = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=self.test_transform)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=self.test_transform)

        n = len(train_aug)
        n_val = int(n * val_frac)
        perm = torch.randperm(n, generator=self.g)
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()

        train_set = Subset(train_aug, train_idx)     # с аугментацией
        val_set = Subset(train_plain, val_idx)       # без аугментации

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train_model(self, epochs: int = 60, lr: float = 0.1, weight_decay: float = 5e-4,
                    momentum: float = 0.9, patience: int = 15, verbose: bool = True):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            train_loss = epoch_loss / n_batches
            val_loss, val_acc = self.evaluate(self.val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if verbose:
                lr_now = scheduler.get_last_lr()[0]
                print(f"epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"val_acc={val_acc:.4f}  lr={lr_now:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"early stopping на эпохе {epoch} (лучший val_loss={best_val:.4f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.train_losses, self.val_losses

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            loss = F.cross_entropy(logits, yb)

            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.shape[0]

        return total_loss / n_batches, correct / total

    def test_accuracy(self) -> float:
        _, acc = self.evaluate(self.test_loader)
        return acc

    def plot_losses(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.train_losses, label='train')
        ax1.plot(self.val_losses, label='val')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend()
        ax1.set_title('Training and validation loss')

        ax2.plot(self.val_accs, label='val acc', color='green')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        ax2.set_title('Validation accuracy')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    m = ResNetFinal()
    m.train_model()
    m.plot_losses()
    print("test accuracy:", m.test_accuracy())