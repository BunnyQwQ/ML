# Reinforcement Learning

Проекты по обучению с подкреплением — от табличных методов (Q-learning, SARSA)
до deep RL (DQN, policy gradient, actor-critic). Каждый проект живёт в своей
подпапке с собственным README.

## Проекты

| Проект | Алгоритм | Среда | Результат |
|--------|----------|-------|-----------|
| [frozen_lake](frozen_lake/) | Табличный Q-learning (ε-greedy, TD) | `FrozenLake-v1` | success rate **0.72** (стохастич.) |
| [cart_pole](cart_pole/) | REINFORCE ± baseline (policy gradient) | `CartPole-v1` | **500** с baseline (без — ~300 за 1000 эп.) |

## Структура

```
RL/
├── <project_name>/
│   ├── README.md      # описание, алгоритм, результаты
│   └── ...            # код
└── README.md          # этот индекс
```
