"""
AlphaZero для Connect Four

Режимы:
    python alphazero_c4.py train        обучение
    python alphazero_c4.py play         игра против обученной модели

"""

import argparse
import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROWS, COLS = 6, 7
ACTION_SIZE = COLS


def initial_board() -> np.ndarray:
    return np.zeros((ROWS, COLS), dtype=np.int8)


def legal_moves(board: np.ndarray) -> np.ndarray:
    return board[0] == 0


def _drop_row(board: np.ndarray, col: int) -> int:
    for row in range(ROWS - 1, -1, -1):
        if board[row, col] == 0:
            return row
    raise ValueError(f"столбец {col} заполнен")


def _check_win(board: np.ndarray, row: int, col: int) -> bool:
    player = board[row, col]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= 4:
            return True
    return False


def next_state(board: np.ndarray, col: int):
    b = board.copy()
    row = _drop_row(b, col)
    b[row, col] = 1
    won = _check_win(b, row, col)
    draw = (not won) and (b != 0).all()
    return -b, won, draw


def encode(board: np.ndarray) -> np.ndarray:
    return np.stack([(board == 1), (board == -1)]).astype(np.float32)


def print_board(board: np.ndarray, current_player: int) -> None:
    disp = board if current_player == 0 else -board
    print()
    for row in range(ROWS):
        line = " |"
        for col in range(COLS):
            v = disp[row, col]
            line += " X" if v == 1 else (" O" if v == -1 else " .")
        print(line + " |")
    print(" +---------------+")
    print("   0 1 2 3 4 5 6\n")


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(x + h)


class AlphaZeroNet(nn.Module):
    def __init__(self, channels: int = 64, n_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        self.p_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * ROWS * COLS, ACTION_SIZE)

        self.v_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * ROWS * COLS, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.blocks(self.stem(x))

        p = F.relu(self.p_bn(self.p_conv(h)))
        p = self.p_fc(p.flatten(1))

        v = F.relu(self.v_bn(self.v_conv(h)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)

        return p, v


@torch.no_grad()
def evaluate(net: AlphaZeroNet, board: np.ndarray, device):
    x = torch.from_numpy(encode(board)).unsqueeze(0).to(device)
    logits, value = net(x)

    policy = F.softmax(logits[0], dim=0).cpu().numpy()

    mask = legal_moves(board)
    policy = policy * mask
    s = policy.sum()
    policy = policy / s if s > 0 else mask / mask.sum()

    return policy, float(value.item())


class Node:
    __slots__ = ("board", "prior", "N", "W", "children", "terminal_value")
    def __init__(self, board: np.ndarray, prior: float = 0.0):
        self.board = board
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children: dict[int, "Node"] = {}
        self.terminal_value: float | None = None

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MCTS:
    def __init__(self, net: AlphaZeroNet, device, c_puct: float = 1.5,
                 dirichlet_alpha: float = 1.0, dirichlet_eps: float = 0.25):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps


    def _select_child(self, node: Node):
        best_score, best_col, best_child = -1e18, -1, None
        sqrt_n = math.sqrt(node.N)

        for col, child in node.children.items():
            q = -child.Q
            u = self.c_puct * child.prior * sqrt_n / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score, best_col, best_child = score, col, child
        return best_col, best_child


    def _expand(self, node: Node) -> float:
        policy, value = evaluate(self.net, node.board, self.device)

        for col in range(ACTION_SIZE):
            if policy[col] <= 0:
                continue
            child_board, won, draw = next_state(node.board, col)
            child = Node(child_board, prior=float(policy[col]))
            if won:
                child.terminal_value = -1.0
            elif draw:
                child.terminal_value = 0.0
            node.children[col] = child

        return value

    def _add_dirichlet_noise(self, root: Node) -> None:
        cols = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(cols))
        for col, nz in zip(cols, noise):
            child = root.children[col]
            child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * nz

    def run(self, board: np.ndarray, n_simulations: int, add_noise: bool = False) -> np.ndarray:
        root = Node(board.copy())
        self._expand(root)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(n_simulations):
            node = root
            path = [root]

            while node.is_expanded and node.terminal_value is None:
                _, node = self._select_child(node)
                path.append(node)
            if node.terminal_value is not None:
                value = node.terminal_value
            else:
                value = self._expand(node)

            for n in reversed(path):
                n.N += 1
                n.W += value
                value = -value

        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        for col, child in root.children.items():
            pi[col] = child.N
        total = pi.sum()
        return pi / total if total > 0 else pi

    def root_stats(self, board: np.ndarray, n_simulations: int):
        root = Node(board.copy())
        self._expand(root)

        for _ in range(n_simulations):
            node, path = root, [root]
            while node.is_expanded and node.terminal_value is None:
                _, node = self._select_child(node)
                path.append(node)
            value = node.terminal_value if node.terminal_value is not None else self._expand(node)
            for n in reversed(path):
                n.N += 1
                n.W += value
                value = -value

        visits = np.zeros(ACTION_SIZE, dtype=np.int64)
        qvals = np.full(ACTION_SIZE, np.nan)
        for col, child in root.children.items():
            visits[col] = child.N
            if child.N > 0:
                qvals[col] = -child.Q
        return visits, qvals


def self_play_game(net: AlphaZeroNet, device, n_simulations: int,
                   temp_moves: int = 10, c_puct: float = 1.5):
    mcts = MCTS(net, device, c_puct=c_puct)
    board = initial_board()
    history = []
    move_count = 0

    while True:
        pi = mcts.run(board, n_simulations, add_noise=True)
        history.append((encode(board), pi.copy()))

        if move_count < temp_moves:
            col = int(np.random.choice(ACTION_SIZE, p=pi))
        else:
            col = int(np.argmax(pi))

        board, won, draw = next_state(board, col)
        move_count += 1

        if won or draw:

            examples = []
            z = 0.0 if draw else 1.0
            for state, pi_saved in reversed(history):
                examples.append((state, pi_saved, z))
                z = -z
            return examples


def train_step(net, optimizer, batch, device):
    states, pis, zs = zip(*batch)
    states = torch.from_numpy(np.stack(states)).to(device)
    pis = torch.from_numpy(np.stack(pis)).to(device)
    zs = torch.tensor(zs, dtype=torch.float32, device=device)

    logits, values = net(states)

    loss_value = F.mse_loss(values, zs)

    log_p = F.log_softmax(logits, dim=1)
    loss_policy = -(pis * log_p).sum(dim=1).mean()

    loss = loss_value + loss_policy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_value.item(), loss_policy.item()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"устройство: {device}")

    net = AlphaZeroNet(channels=args.channels, n_blocks=args.blocks).to(device)
    if args.resume and os.path.exists(args.model):
        net.load_state_dict(torch.load(args.model, map_location=device))
        print(f"загружены веса из {args.model}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    buffer = deque(maxlen=args.buffer_size)

    for it in range(args.iterations):
        net.eval()
        n_new = 0
        for g in range(args.games):
            examples = self_play_game(net, device, args.simulations, temp_moves=args.temp_moves, c_puct=args.c_puct)
            buffer.extend(examples)
            n_new += len(examples)
            print(f"\r  итерация {it + 1}: партия {g + 1}/{args.games}  позиций {n_new}", end="", flush=True)
        print()

        if len(buffer) < args.batch_size:
            continue

        net.train()
        tot_l = tot_v = tot_p = 0.0
        for step in range(args.train_steps):
            batch = random.sample(buffer, args.batch_size)
            l, lv, lp = train_step(net, optimizer, batch, device)
            tot_l += l
            tot_v += lv
            tot_p += lp

        n = args.train_steps
        print(f"  итерация {it + 1}: loss={tot_l / n:.4f} value={tot_v / n:.4f}  policy={tot_p / n:.4f} буфер={len(buffer)}")

        torch.save(net.state_dict(), args.model)

    print(f"\nмодель сохранена в {args.model}")


def read_human_move(board: np.ndarray) -> int:
    mask = legal_moves(board)
    while True:
        raw = input("твой ход (0-6, q — выход): ").strip()
        if raw.lower() == "q":
            return -1
        if not raw.isdigit():
            print("  нужно число от 0 до 6")
            continue
        col = int(raw)
        if not (0 <= col < COLS):
            print("  столбец вне диапазона")
        elif not mask[col]:
            print("  столбец заполнен")
        else:
            return col


def play_vs_human(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = AlphaZeroNet(channels=args.channels, n_blocks=args.blocks).to(device)
    if os.path.exists(args.model):
        net.load_state_dict(torch.load(args.model, map_location=device))
        print(f"загружена модель {args.model}")
    else:
        print("модель не найдена — сеть со случайными весами, играть будет слабо")
    net.eval()

    mcts = MCTS(net, device, c_puct=args.c_puct)

    while True:
        side = input("играть за X (первым) или O (вторым)? [x/o]: ").strip().lower()
        if side in ("x", "o"):
            human = 0 if side == "x" else 1
            break

    board = initial_board()
    current = 0
    print_board(board, current)

    while True:
        if current == human:
            col = read_human_move(board)
            if col == -1:
                print("выход")
                return
        else:
            visits, qvals = mcts.root_stats(board, args.simulations)
            col = int(np.argmax(visits))

            print(f"\nAlphaZero выбрал столбец {col}")
            print("  столбец:  " + "".join(f"{c:>8}" for c in range(COLS)))
            print("  посещений:" + "".join(f"{v:>8}" for v in visits))
            print("  оценка:   " + "".join(f"{q:>8.2f}" if not np.isnan(q) else f"{'-':>8}" for q in qvals))
            print()

        board, won, draw = next_state(board, col)
        winner = current
        current ^= 1
        print_board(board, current)

        if won:
            print("ты победил" if winner == human else "AlphaZero победил")
            return
        if draw:
            print("ничья")
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "play"])

    # архитектура
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--model", type=str, default="alphazero_c4.pt")

    # поиск
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--c-puct", type=float, default=1.5)

    # обучение
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temp-moves", type=int, default=10)
    p.add_argument("--resume", action="store_true")

    args = p.parse_args()

    if args.mode == "train":
        train(args)
    else:
        play_vs_human(args)


if __name__ == "__main__":
    main()