import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import time

class PolicyNet(nn.Module):
    ''' Agent '''
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    ''' Critic '''
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ReinforceAgent:
    def __init__(
        self,
        env_name: str = 'CartPole-v1',
        seed: int = 42,
        gamma: float = 0.99,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        use_baseline: bool = True,
        normalize_returns: bool = False,
        hidden: int = 128,
    ):
        self.env = gym.make(env_name)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        state_dim = self.env.observation_space.shape[0] # 4
        n_actions = self.env.action_space.n # 2

        self.actor = PolicyNet(state_dim, n_actions, hidden)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.use_baseline = use_baseline
        if use_baseline:
            self.critic = ValueNet(state_dim, hidden)
            self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.normalize_returns = normalize_returns
        self.rewards_history: list[float] = []

    @torch.no_grad()
    def choose_action(self, state: np.ndarray) -> int:
        s = torch.tensor(state, dtype=torch.float32)
        logits = self.actor(s)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item())

    def compute_returns(self, rewards: list[float]) -> torch.Tensor:
        returns = []
        G = 0.0
        for r in rewards[::-1]:
            G = r + self.gamma * G
            returns.append(G)
        return torch.tensor(returns[::-1], dtype=torch.float32)

    def collect_episode(self):
        state, _ = self.env.reset(seed=int(self.rng.integers(0, 1_000_000)))
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        return (
            torch.tensor(np.array(states), dtype=torch.float32), # (T, state_dim)
            torch.tensor(actions, dtype=torch.long), # (T)
            rewards # list[float]
        )

    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: list[float]):
        returns = self.compute_returns(rewards)

        if self.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        if self.use_baseline:
            values = self.critic(states)
            advantages = returns - values.detach()
        else:
            values = None
            advantages = returns

        logits = self.actor(states)
        log_probs_all = F.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss_actor = -(log_probs * advantages).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        if self.use_baseline:
            loss_critic = F.mse_loss(values, returns)
            self.opt_critic.zero_grad()
            loss_critic.backward()
            self.opt_critic.step()

    def train(self, episodes: int = 1000, verbose: bool = True):
        for ep in range(episodes):
            states, actions, rewards = self.collect_episode()
            self.update(states, actions, rewards)

            self.rewards_history.append(sum(rewards))

            if verbose and (ep + 1) % 50 == 0:
                recent = np.mean(self.rewards_history[-50:])
                print(f"эпизод {ep + 1}: средняя награда (50) = {recent:.1f}")

            if len(self.rewards_history) >= 100 and np.mean(self.rewards_history[-100:]) >= 475:
                print(f"решено на эпизоде {ep + 1}")
                break

        return self.rewards_history

    @torch.no_grad()
    def evaluate(self, episodes: int = 20) -> float:
        total = 0.0
        for _ in range(episodes):
            state, _ = self.env.reset(seed=int(self.rng.integers(0, 1_000_000)))
            done = False
            while not done:
                s = torch.tensor(state, dtype=torch.float32)
                action = int(torch.argmax(self.actor(s)).item())
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total += reward
        return total / episodes

    def plot_progress(self, window: int = 50):
        smoothed = np.convolve(self.rewards_history, np.ones(window) / window, mode='valid')
        plt.figure(figsize=(8, 4))
        plt.plot(self.rewards_history)
        plt.plot(range(window - 1, len(self.rewards_history)), smoothed, label=f'скользящее среднее ({window})')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.show()


    @torch.no_grad()
    def watch(self, episodes: int = 3, deterministic: bool = True, delay: float = 0.02):
        env = gym.make('CartPole-v1', render_mode='human')
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                s = torch.tensor(state, dtype=torch.float32)
                logits = self.actor(s)
                if deterministic:
                    action = int(torch.argmax(logits).item())
                else:
                    action = int(Categorical(logits=logits).sample().item())
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
                time.sleep(delay)
            print(f"эпизод {ep + 1}: награда {total:.0f}")
            time.sleep(1.0)
        env.close()


if __name__ == '__main__':
    agent = ReinforceAgent(baseline=True)
    agent.watch()
    agent.train(episodes=1000)
    agent.watch()
    agent.plot_progress()
    print("eval:", agent.evaluate())