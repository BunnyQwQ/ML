# Reinforcement Learning

Проекты по обучению с подкреплением — от табличных методов (Q-learning) через
deep RL (policy gradient, PPO) до поиска по дереву и self-play (MCTS, AlphaZero).
Каждый проект живёт в своей подпапке с собственным README.

## Проекты

| Проект | Алгоритм | Среда | Результат |
|--------|----------|-------|-----------|
| [frozen_lake](frozen_lake/) | Табличный Q-learning (ε-greedy, TD) | `FrozenLake-v1` | success rate **0.72** (стохастич.) |
| [cart_pole](cart_pole/) | REINFORCE (± baseline) и PPO (GAE, clipping) | `CartPole-v1` | PPO и REINFORCE+baseline → **500**; REINFORCE без baseline → ~300 |
| [connect_four](connect_four/) | Классический MCTS (C++) и AlphaZero (self-play, PyTorch) | Connect Four (своя реализация) | обученная модель в комплекте, играет против человека |

## Структура

```
RL/
├── <project_name>/
│   ├── README.md      # описание, алгоритм, результаты
│   └── ...            # код
└── README.md          # этот индекс
```
