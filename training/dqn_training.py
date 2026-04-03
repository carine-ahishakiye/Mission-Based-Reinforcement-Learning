import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import UlinziEnv

os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        self._current_length += 1
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0
        return True


HYPERPARAMETER_RUNS = [
    {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 64,  "buffer_size": 10000, "exploration_fraction": 0.2, "exploration_final_eps": 0.05, "target_update_interval": 500,  "train_freq": 4},
    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 64,  "buffer_size": 50000, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 1000, "train_freq": 4},
    {"learning_rate": 1e-4, "gamma": 0.95, "batch_size": 32,  "buffer_size": 10000, "exploration_fraction": 0.4, "exploration_final_eps": 0.1,  "target_update_interval": 500,  "train_freq": 4},
    {"learning_rate": 1e-3, "gamma": 0.90, "batch_size": 128, "buffer_size": 20000, "exploration_fraction": 0.2, "exploration_final_eps": 0.05, "target_update_interval": 250,  "train_freq": 8},
    {"learning_rate": 2e-4, "gamma": 0.99, "batch_size": 64,  "buffer_size": 50000, "exploration_fraction": 0.5, "exploration_final_eps": 0.02, "target_update_interval": 1000, "train_freq": 4},
    {"learning_rate": 5e-3, "gamma": 0.95, "batch_size": 32,  "buffer_size": 10000, "exploration_fraction": 0.3, "exploration_final_eps": 0.1,  "target_update_interval": 500,  "train_freq": 4},
    {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 128, "buffer_size": 100000,"exploration_fraction": 0.2, "exploration_final_eps": 0.05, "target_update_interval": 2000, "train_freq": 4},
    {"learning_rate": 3e-4, "gamma": 0.97, "batch_size": 64,  "buffer_size": 20000, "exploration_fraction": 0.35,"exploration_final_eps": 0.05, "target_update_interval": 750,  "train_freq": 8},
    {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 256, "buffer_size": 50000, "exploration_fraction": 0.25,"exploration_final_eps": 0.01, "target_update_interval": 1000, "train_freq": 4},
    {"learning_rate": 6e-4, "gamma": 0.98, "batch_size": 64,  "buffer_size": 30000, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 600,  "train_freq": 4},
]

TIMESTEPS = 80000
EVAL_EPISODES = 20


def make_env(seed=0):
    env = UlinziEnv(max_steps=48)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_dqn_run(run_idx, params):
    print(f"\n{'='*60}")
    print(f"  DQN Run {run_idx + 1}/10")
    print(f"  LR={params['learning_rate']} | gamma={params['gamma']} | batch={params['batch_size']}")
    print(f"{'='*60}")

    env = make_env(seed=run_idx)
    eval_env = make_env(seed=100 + run_idx)

    callback = RewardLoggerCallback()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        buffer_size=params["buffer_size"],
        exploration_fraction=params["exploration_fraction"],
        exploration_final_eps=params["exploration_final_eps"],
        target_update_interval=params["target_update_interval"],
        train_freq=params["train_freq"],
        learning_starts=1000,
        verbose=0,
        seed=run_idx,
    )

    model.learn(total_timesteps=TIMESTEPS, callback=callback, progress_bar=False)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)

    env.close()
    eval_env.close()

    result = {
        "run": run_idx + 1,
        "mean_reward": round(float(mean_reward), 3),
        "std_reward": round(float(std_reward), 3),
        "episode_rewards": callback.episode_rewards,
        **params,
    }

    print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    return model, result


def plot_dqn_results(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Hyperparameter Experiment Results — ULINZI", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for r in all_results:
        smoothed = np.convolve(r["episode_rewards"], np.ones(10) / 10, mode="valid") if len(r["episode_rewards"]) >= 10 else r["episode_rewards"]
        ax.plot(smoothed, alpha=0.7, label=f"Run {r['run']} (lr={r['learning_rate']:.0e})")
    ax.set_title("Cumulative Reward per Episode (Smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    runs = [r["run"] for r in all_results]
    means = [r["mean_reward"] for r in all_results]
    stds = [r["std_reward"] for r in all_results]
    bars = ax.bar(runs, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_title("Mean Evaluation Reward per Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Mean Reward")
    ax.set_xticks(runs)
    ax.grid(True, alpha=0.3, axis="y")
    best_idx = int(np.argmax(means))
    bars[best_idx].set_color("gold")
    ax.annotate("Best", xy=(runs[best_idx], means[best_idx] + stds[best_idx] + 0.2), ha="center", fontsize=9, color="darkorange")

    ax = axes[1, 0]
    lrs = [r["learning_rate"] for r in all_results]
    ax.scatter(lrs, means, c=means, cmap="RdYlGn", s=100, zorder=3)
    ax.set_xscale("log")
    ax.set_title("Learning Rate vs Mean Reward")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    gammas = [r["gamma"] for r in all_results]
    ax.scatter(gammas, means, c=means, cmap="RdYlGn", s=100, zorder=3)
    ax.set_title("Discount Factor (gamma) vs Mean Reward")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "logs/dqn/dqn_hyperparameter_results.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {path}")


def main():
    all_results = []
    best_reward = -np.inf
    best_model = None
    best_run = 0

    for i, params in enumerate(HYPERPARAMETER_RUNS):
        model, result = train_dqn_run(i, params)
        all_results.append(result)

        if result["mean_reward"] > best_reward:
            best_reward = result["mean_reward"]
            best_model = model
            best_run = i + 1

    print(f"\n{'='*60}")
    print(f"  Best DQN Run: {best_run} with Mean Reward: {best_reward:.3f}")
    print(f"{'='*60}")

    best_model.save("models/dqn/dqn_best")
    print("  Best model saved to models/dqn/dqn_best.zip")

    summary = [
        {k: v for k, v in r.items() if k != "episode_rewards"}
        for r in all_results
    ]
    with open("logs/dqn/dqn_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_dqn_results(all_results)

    print("\n  DQN Summary Table:")
    print(f"  {'Run':>3} | {'LR':>8} | {'Gamma':>5} | {'Batch':>5} | {'Mean Reward':>12} | {'Std':>6}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['run']:>3} | {r['learning_rate']:>8.1e} | {r['gamma']:>5.2f} | {r['batch_size']:>5} | {r['mean_reward']:>12.3f} | {r['std_reward']:>6.3f}")


if __name__ == "__main__":
    main()