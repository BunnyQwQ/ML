import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import time

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)


class PPOAgent:
    def __init__(
            self,
            env_name: str = "CartPole-v1",
            num_envs: int = 8,
            n_steps: int = 128,
            gamma: float = 0.99,
            lr: float = 1e-3,
            seed: int = 42,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            n_epochs: int = 10,
            minibatch_size: int = 256,
            cr_coef : float = 0.5,
            ent_coef: float = 0.01,
            hidden: int = 64,
            max_grad_norm: float = 0.5,
    ):
        self.envs = gym.make_vec(env_name, num_envs=num_envs)
        torch.manual_seed(seed)

        self.state_dim = self.envs.single_observation_space.shape[0]
        n_actions = self.envs.single_action_space.n

        self.net = ActorCritic(self.state_dim, n_actions, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self.num_envs = num_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.cr_coef = cr_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.state, _ = self.envs.reset(seed=seed)
        self.episode_returns: list[float] = []
        self._running = np.zeros(num_envs)

    @torch.no_grad()
    def collect_rollout(self):
        states = torch.zeros(self.n_steps, self.num_envs, self.state_dim, dtype=torch.float)
        actions = torch.zeros(self.n_steps, self.num_envs, dtype=torch.long)
        log_probs = torch.zeros(self.n_steps, self.num_envs)
        values = torch.zeros(self.n_steps, self.num_envs)
        rewards = torch.zeros(self.n_steps, self.num_envs)
        dones = torch.zeros(self.n_steps, self.num_envs)
        for t in range(self.n_steps):
            s = torch.tensor(self.state, dtype=torch.float)
            logits, value = self.net(s)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            states[t] = s
            actions[t] = action
            log_probs[t] = dist.log_prob(action)
            values[t] = value

            next_state, reward, term, trunc, _ = self.envs.step(action.numpy())
            done = np.logical_or(term, trunc)

            rewards[t] = torch.tensor(reward, dtype=torch.float32)
            dones[t] = torch.tensor(done, dtype=torch.float32)

            self._running += reward

            for i in range(self.num_envs):
                if done[i]:
                    self.episode_returns.append(self._running[i])
                    self._running[i] = 0.0

            self.state = next_state
        _, last_value = self.net(torch.tensor(self.state, dtype=torch.float32))

        return states, actions, log_probs, values, rewards, dones, last_value

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                next_nonterminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_nonterminal = 1.0 - dones[t]

            td = rewards[t] - values[t] + self.gamma * next_nonterminal * next_value
            last_gae = td + self.gae_lambda * self.gamma * last_gae * next_nonterminal
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, log_probs_old, advantages, returns):
        b_states = states.reshape(-1, states.shape[-1])
        b_actions = actions.reshape(-1)
        b_log_probs_old = log_probs_old.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        batch_size = b_states.shape[0]
        indices = np.arange(batch_size)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]

                mb_states = b_states[mb]
                mb_actions = b_actions[mb]
                mb_log_probs_old = b_log_probs_old[mb]
                mb_advantages = b_advantages[mb]
                mb_returns = b_returns[mb]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                logits, values_new = self.net(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs_new = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs_new - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clip(ratio,1. - self.clip_eps, 1. + self.clip_eps) * mb_advantages
                loss_actor = -torch.min(surr1, surr2).mean()

                loss_critic = F.mse_loss(values_new, mb_returns)

                loss = loss_actor + self.cr_coef * loss_critic - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

    def train(self, total_steps: int = 200_000, verbose: bool = True):
        steps_per_iter = self.n_steps * self.num_envs
        n_iters = total_steps // steps_per_iter

        for it in range(n_iters):
            states, actions, log_probs, values, rewards, dones, last_value = self.collect_rollout()
            advantages, returns = self.compute_gae(rewards, values, dones, last_value)
            self.update(states, actions, log_probs, advantages, returns)

            if verbose and (it + 1) % 10 == 0 and self.episode_returns:
                recent = np.mean(self.episode_returns[-50:])
                done_steps = (it + 1) * steps_per_iter
                print(f"итерация {it + 1}  шагов среды {done_steps}  "
                      f"средняя награда (50) = {recent:.1f}")

        return self.episode_returns

    @torch.no_grad()
    def evaluate(self, episodes: int = 20) -> float:
        env = gym.make('CartPole-v1')
        total = 0.0
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            while not done:
                s = torch.tensor(state, dtype=torch.float32)
                logits, _ = self.net(s)
                action = int(torch.argmax(logits).item())
                state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                total += reward
        env.close()
        return total / episodes

    def plot_progress(self, window: int = 50):
        if len(self.episode_returns) < window:
            print("мало эпизодов для графика")
            return
        smoothed = np.convolve(self.episode_returns, np.ones(window) / window, mode='valid')
        plt.figure(figsize=(8, 4))
        plt.plot(self.episode_returns, alpha=0.3, label='эпизод')
        plt.plot(range(window - 1, len(self.episode_returns)), smoothed,
                 label=f'скользящее среднее ({window})')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.title('PPO на CartPole')
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
                logits, _ = self.net(s)
                if deterministic:
                    action = int(torch.argmax(logits).item())
                else:
                    action = int(Categorical(logits=logits).sample().item())
                state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                total += reward
                time.sleep(delay)
            print(f"эпизод {ep + 1}: награда {total:.0f}")
            time.sleep(1.0)
        env.close()

if __name__ == '__main__':
    agent = PPOAgent()
    agent.train(total_steps=200_000)
    agent.watch()
    agent.watch(deterministic=False)
    agent.plot_progress()
    print("eval:", agent.evaluate())