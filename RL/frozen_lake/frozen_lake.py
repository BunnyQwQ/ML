import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(
        self,
        env_name: str = 'FrozenLake-v1',
        is_slippery: bool = False,
        seed: int = 43,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.99999,
    ):
        self.env = gym.make(env_name, is_slippery=is_slippery)
        self.rng = np.random.default_rng(seed)

        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rewards_history: list[float] = []
        self.epsilon_history: list[float] = []

    def choose_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train(self, episodes: int = 5000, verbose: bool = True):
        for episode in range(episodes):
            state, _ = self.env.reset(seed=int(self.rng.integers(0, 1_000_000)))
            total_reward = 0.0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            self.rewards_history.append(total_reward)
            self.epsilon_history.append(self.epsilon)

            if verbose and (episode + 1) % 500 == 0:
                recent = np.mean(self.rewards_history[-500:])
                print(f"эпизод {episode + 1}: success_rate(500)={recent:.3f}  eps={self.epsilon:.3f}")

        return self.rewards_history

    def evaluate(self, episodes: int = 100) -> float:
        successes = 0
        for _ in range(episodes):
            state, _ = self.env.reset(seed=int(self.rng.integers(0, 1_000_000)))
            done = False
            while not done:
                action = np.argmax(self.Q[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                if reward > 0:
                    successes += 1
        return successes / episodes

    def plot_progress(self, window: int = 100):
        smoothed = np.convolve(
            self.rewards_history, np.ones(window) / window, mode='valid'
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(smoothed)
        ax1.set_xlabel('episode')
        ax1.set_ylabel(f'success rate (окно {window})')
        ax1.set_title('Обучение')

        ax2.plot(self.epsilon_history, color='orange')
        ax2.set_xlabel('episode')
        ax2.set_ylabel('epsilon')
        ax2.set_title('Затухание исследования')

        plt.tight_layout()
        plt.show()

    def print_policy(self):
        arrows = ['<', 'v', '>', '^']
        desc = self.env.unwrapped.desc.astype(str)
        n = desc.shape[0]
        best = np.argmax(self.Q, axis=1)

        print("\nВыученная политика:")
        for row in range(n):
            line = []
            for col in range(n):
                cell = desc[row, col]
                if cell == 'H':
                    line.append('H')
                elif cell == 'G':
                    line.append('G')
                else:
                    line.append(arrows[best[row * n + col]])
            print(' '.join(line))

    def print_q_table(self):
        print("\nQ-таблица (строка = состояние, столбец = действие):")
        print("      <       v       >       ^")
        for state in range(self.n_states):
            row = '  '.join(f"{v:6.3f}" for v in self.Q[state])
            print(f"{state:2d}  {row}")


if __name__ == '__main__':
    agent = QLearningAgent(is_slippery=True)
    agent.train(episodes=50000)
    agent.plot_progress()
    agent.print_policy()
    agent.print_q_table()
    print("\nsuccess rate (eval):", agent.evaluate(episodes=100))
