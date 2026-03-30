import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import environment.custom_env  # noqa: F401
import gymnasium as gym

MODELS_DIR  = os.path.join("models", "pg")
RESULTS_DIR = os.path.join("models", "pg", "results")
PLOTS_DIR   = os.path.join("models", "pg", "plots")
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 50_000
EVAL_EPISODES   = 20


def make_env():
    return Monitor(gym.make("WildlifeConflict-v0"))


class PGLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.episode_rewards = []
        self.entropy_log     = []
        self._current_ep_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done   = self.locals.get("dones",   [False])[0]
        self._current_ep_reward += reward
        if done:
            self.episode_rewards.append(self._current_ep_reward)
            self._current_ep_reward = 0.0

        # Capture entropy from SB3 logger if available
        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                ent = self.model.logger.name_to_value.get("train/entropy_loss", None)
                if ent is not None:
                    self.entropy_log.append(float(ent))
            except Exception:
                pass
        return True

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def _run_reinforce_once(hp, env, eval_env, obs_dim, n_actions):
    """
    One REINFORCE training run.
    Returns (mean_reward, std_reward, episode_rewards, entropy_log).
    """
    policy    = PolicyNet(obs_dim, n_actions, hidden=hp["hidden_size"])
    optimizer = optim.Adam(policy.parameters(), lr=hp["learning_rate"])

    episode_rewards = []
    entropy_log     = []
    total_steps     = 0

    while total_steps < TOTAL_TIMESTEPS:
        obs, _    = env.reset()
        done      = False
        log_probs = []
        rewards   = []
        entropies = []

        while not done:
            obs_t  = torch.FloatTensor(obs).unsqueeze(0)
            probs  = policy(obs_t)
            dist   = Categorical(probs)
            action = dist.sample()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            entropies.append(dist.entropy().item())
            total_steps += 1

        # Discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + hp["gamma"] * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -torch.stack(
            [lp * Gt for lp, Gt in zip(log_probs, returns)]
        ).sum()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), hp["max_grad_norm"])
        optimizer.step()

        episode_rewards.append(sum(rewards))
        entropy_log.append(np.mean(entropies))

    # Greedy evaluation
    eval_ep_rewards = []
    for _ in range(EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done, ep_r = False, 0.0
        while not done:
            with torch.no_grad():
                probs  = policy(torch.FloatTensor(obs).unsqueeze(0))
                action = torch.argmax(probs).item()
            obs, r, term, trunc, _ = eval_env.step(action)
            ep_r += r
            done = term or trunc
        eval_ep_rewards.append(ep_r)

    return (float(np.mean(eval_ep_rewards)),
            float(np.std(eval_ep_rewards)),
            episode_rewards,
            entropy_log)


# Hyperparameter grids 

REINFORCE_GRID = [
    {"run": 1,  "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64,  "max_grad_norm": 0.5},
    {"run": 2,  "learning_rate": 5e-4, "gamma": 0.99, "hidden_size": 64,  "max_grad_norm": 0.5},
    {"run": 3,  "learning_rate": 1e-4, "gamma": 0.99, "hidden_size": 128, "max_grad_norm": 0.5},
    {"run": 4,  "learning_rate": 1e-3, "gamma": 0.95, "hidden_size": 64,  "max_grad_norm": 1.0},
    {"run": 5,  "learning_rate": 1e-3, "gamma": 0.90, "hidden_size": 64,  "max_grad_norm": 0.5},
    {"run": 6,  "learning_rate": 2e-3, "gamma": 0.99, "hidden_size": 64,  "max_grad_norm": 0.5},
    {"run": 7,  "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 256, "max_grad_norm": 0.5},
    {"run": 8,  "learning_rate": 5e-4, "gamma": 0.95, "hidden_size": 128, "max_grad_norm": 1.0},
    {"run": 9,  "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64,  "max_grad_norm": 2.0},
    {"run": 10, "learning_rate": 3e-4, "gamma": 0.97, "hidden_size": 128, "max_grad_norm": 0.5},
]

PPO_GRID = [
    {"run": 1,  "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.00, "gae_lambda": 0.95},
    {"run": 2,  "learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01, "gae_lambda": 0.95},
    {"run": 3,  "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64,  "n_epochs": 5,  "clip_range": 0.2, "ent_coef": 0.00, "gae_lambda": 0.95},
    {"run": 4,  "learning_rate": 3e-4, "gamma": 0.95, "n_steps": 2048, "batch_size": 128, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.00, "gae_lambda": 0.90},
    {"run": 5,  "learning_rate": 5e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "clip_range": 0.3, "ent_coef": 0.01, "gae_lambda": 0.95},
    {"run": 6,  "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512,  "batch_size": 64,  "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.00, "gae_lambda": 0.95},
    {"run": 7,  "learning_rate": 1e-3, "gamma": 0.99, "n_steps": 2048, "batch_size": 256, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.00, "gae_lambda": 0.95},
    {"run": 8,  "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "n_epochs": 20, "clip_range": 0.1, "ent_coef": 0.00, "gae_lambda": 0.95},
    {"run": 9,  "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 4096, "batch_size": 64,  "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.05, "gae_lambda": 0.95},
    {"run": 10, "learning_rate": 2e-4, "gamma": 0.97, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01, "gae_lambda": 0.98},
]

A2C_GRID = [
    {"run": 1,  "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.00, "vf_coef": 0.50, "max_grad_norm": 0.5, "gae_lambda": 1.00},
    {"run": 2,  "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.01, "vf_coef": 0.50, "max_grad_norm": 0.5, "gae_lambda": 1.00},
    {"run": 3,  "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.00, "vf_coef": 0.50, "max_grad_norm": 0.5, "gae_lambda": 0.95},
    {"run": 4,  "learning_rate": 7e-4, "gamma": 0.95, "n_steps": 5,  "ent_coef": 0.00, "vf_coef": 0.50, "max_grad_norm": 0.5, "gae_lambda": 1.00},
    {"run": 5,  "learning_rate": 1e-3, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.00, "vf_coef": 0.50, "max_grad_norm": 0.5, "gae_lambda": 1.00},
    {"run": 6,  "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 20, "ent_coef": 0.01, "vf_coef": 0.50, "max_grad_norm": 0.5, "gae_lambda": 0.95},
    {"run": 7,  "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.00, "vf_coef": 1.00, "max_grad_norm": 0.5, "gae_lambda": 1.00},
    {"run": 8,  "learning_rate": 5e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.05, "vf_coef": 0.50, "max_grad_norm": 1.0, "gae_lambda": 1.00},
    {"run": 9,  "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.00, "vf_coef": 0.25, "max_grad_norm": 0.5, "gae_lambda": 1.00},
    {"run": 10, "learning_rate": 2e-4, "gamma": 0.97, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.50, "max_grad_norm": 0.5, "gae_lambda": 0.98},
]


#  Shared plotting helpers

def _plot_reward_curves(rewards_per_run: dict, title: str, path: str):
    n    = len(rewards_per_run)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
    axes = axes.flatten()

    for i, (label, rewards) in enumerate(rewards_per_run.items()):
        cumulative = np.cumsum(rewards)
        axes[i].plot(cumulative, color="darkorange", linewidth=1.2)
        axes[i].set_title(label, fontsize=8)
        axes[i].set_xlabel("Episode", fontsize=7)
        axes[i].set_ylabel("Cumulative Reward", fontsize=7)
        axes[i].tick_params(labelsize=6)
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved reward curves → {path}")


def _plot_entropy_curves(entropy_per_run: dict, title: str, path: str):
    n    = len(entropy_per_run)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
    axes = axes.flatten()

    for i, (label, entropy) in enumerate(entropy_per_run.items()):
        axes[i].plot(entropy, color="mediumseagreen", linewidth=1.2)
        axes[i].set_title(label, fontsize=8)
        axes[i].set_xlabel("Episode / Update", fontsize=7)
        axes[i].set_ylabel("Entropy", fontsize=7)
        axes[i].tick_params(labelsize=6)
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved entropy curves → {path}")


def _plot_convergence(df: pd.DataFrame, algo: str, path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["green" if r == df["mean_reward"].max() else "steelblue"
               for r in df["mean_reward"]]
    ax.bar(df["run"].astype(str), df["mean_reward"],
           color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(df["run"].astype(str), df["mean_reward"],
                yerr=df["std_reward"], fmt="none",
                color="black", capsize=4, linewidth=1)
    ax.set_xlabel("Run", fontsize=11)
    ax.set_ylabel("Mean Eval Reward", fontsize=11)
    ax.set_title(f"{algo} — Convergence Across Hyperparameter Runs\n"
                 "(green = best run)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved convergence plot → {path}")


# Training runners

def train_reinforce():
    print("  REINFORCE Hyperparameter")

    env       = make_env()
    eval_env  = make_env()
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    results          = []
    best_mean_reward = -np.inf
    best_run_id      = None
    rewards_per_run  = {}
    entropy_per_run  = {}

    for hp in REINFORCE_GRID:
        run_id = hp["run"]
        print(f"\n[Run {run_id:02d}/10]  lr={hp['learning_rate']}  "
              f"γ={hp['gamma']}  hidden={hp['hidden_size']}  "
              f"grad_clip={hp['max_grad_norm']}")

        mean_r, std_r, ep_rewards, entropy = _run_reinforce_once(
            hp, env, eval_env, obs_dim, n_actions
        )
        print(f"  → Mean reward: {mean_r:.2f} ± {std_r:.2f}")

        label = f"Run {run_id:02d}\nlr={hp['learning_rate']}, h={hp['hidden_size']}"
        rewards_per_run[label] = ep_rewards
        entropy_per_run[label] = entropy

        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            best_run_id      = run_id

        results.append({
            "run": run_id, "learning_rate": hp["learning_rate"],
            "gamma": hp["gamma"], "hidden_size": hp["hidden_size"],
            "max_grad_norm": hp["max_grad_norm"],
            "mean_reward": round(mean_r, 3), "std_reward": round(std_r, 3),
        })

    env.close()
    eval_env.close()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "reinforce_results.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "reinforce_summary.json"), "w") as f:
        json.dump({"best_run": best_run_id,
                   "best_mean_reward": round(best_mean_reward, 3)}, f, indent=2)

    _plot_reward_curves(rewards_per_run,
                        "REINFORCE — Cumulative Reward Curves (All Runs)",
                        os.path.join(PLOTS_DIR, "reinforce_reward_curves.png"))
    _plot_entropy_curves(entropy_per_run,
                         "REINFORCE — Entropy per Episode (All Runs)",
                         os.path.join(PLOTS_DIR, "reinforce_entropy_curves.png"))
    _plot_convergence(df, "REINFORCE",
                      os.path.join(PLOTS_DIR, "reinforce_convergence.png"))

    print(f"\n  Best REINFORCE → Run {best_run_id}  "
          f"|  Mean reward: {best_mean_reward:.2f}")
    return df


def train_ppo():
    print("  PPO Hyperparameter ")

    results          = []
    best_mean_reward = -np.inf
    best_run_id      = None
    rewards_per_run  = {}
    entropy_per_run  = {}

    for hp in PPO_GRID:
        run_id = hp["run"]
        print(f"\n[Run {run_id:02d}/10]  lr={hp['learning_rate']}  "
              f"γ={hp['gamma']}  n_steps={hp['n_steps']}  "
              f"clip={hp['clip_range']}  ent_coef={hp['ent_coef']}")

        env      = make_env()
        eval_env = make_env()
        cb       = PGLoggerCallback()

        model = PPO(
            policy        = "MlpPolicy",
            env           = env,
            learning_rate = hp["learning_rate"],
            gamma         = hp["gamma"],
            n_steps       = hp["n_steps"],
            batch_size    = hp["batch_size"],
            n_epochs      = hp["n_epochs"],
            clip_range    = hp["clip_range"],
            ent_coef      = hp["ent_coef"],
            gae_lambda    = hp["gae_lambda"],
            verbose       = 0,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=cb, progress_bar=False)

        mean_r, std_r = evaluate_policy(
            model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
        )
        print(f"  → Mean reward: {mean_r:.2f} ± {std_r:.2f}")

        label = f"Run {run_id:02d}\nlr={hp['learning_rate']}, clip={hp['clip_range']}"
        rewards_per_run[label] = cb.episode_rewards
        entropy_per_run[label] = cb.entropy_log

        model.save(os.path.join(MODELS_DIR, f"ppo_run{run_id:02d}"))
        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            best_run_id      = run_id
            model.save(os.path.join(MODELS_DIR, "ppo_best"))

        results.append({
            "run": run_id, "learning_rate": hp["learning_rate"],
            "gamma": hp["gamma"], "n_steps": hp["n_steps"],
            "batch_size": hp["batch_size"], "n_epochs": hp["n_epochs"],
            "clip_range": hp["clip_range"], "ent_coef": hp["ent_coef"],
            "gae_lambda": hp["gae_lambda"],
            "mean_reward": round(mean_r, 3), "std_reward": round(std_r, 3),
        })
        env.close()
        eval_env.close()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "ppo_results.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "ppo_summary.json"), "w") as f:
        json.dump({"best_run": best_run_id,
                   "best_mean_reward": round(best_mean_reward, 3)}, f, indent=2)

    _plot_reward_curves(rewards_per_run,
                        "PPO — Cumulative Reward Curves (All Runs)",
                        os.path.join(PLOTS_DIR, "ppo_reward_curves.png"))
    _plot_entropy_curves(entropy_per_run,
                         "PPO — Entropy Loss per Update (All Runs)",
                         os.path.join(PLOTS_DIR, "ppo_entropy_curves.png"))
    _plot_convergence(df, "PPO",
                      os.path.join(PLOTS_DIR, "ppo_convergence.png"))

    print(f"\n  Best PPO → Run {best_run_id}  "
          f"|  Mean reward: {best_mean_reward:.2f}")
    return df


def train_a2c():
    print("  A2C Hyperparameter ")

    results          = []
    best_mean_reward = -np.inf
    best_run_id      = None
    rewards_per_run  = {}
    entropy_per_run  = {}

    for hp in A2C_GRID:
        run_id = hp["run"]
        print(f"\n[Run {run_id:02d}/10]  lr={hp['learning_rate']}  "
              f"γ={hp['gamma']}  n_steps={hp['n_steps']}  "
              f"ent={hp['ent_coef']}  vf={hp['vf_coef']}")

        env      = make_env()
        eval_env = make_env()
        cb       = PGLoggerCallback()

        model = A2C(
            policy        = "MlpPolicy",
            env           = env,
            learning_rate = hp["learning_rate"],
            gamma         = hp["gamma"],
            n_steps       = hp["n_steps"],
            ent_coef      = hp["ent_coef"],
            vf_coef       = hp["vf_coef"],
            max_grad_norm = hp["max_grad_norm"],
            gae_lambda    = hp["gae_lambda"],
            verbose       = 0,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=cb, progress_bar=False)

        mean_r, std_r = evaluate_policy(
            model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
        )
        print(f"  → Mean reward: {mean_r:.2f} ± {std_r:.2f}")

        label = f"Run {run_id:02d}\nlr={hp['learning_rate']}, vf={hp['vf_coef']}"
        rewards_per_run[label] = cb.episode_rewards
        entropy_per_run[label] = cb.entropy_log

        model.save(os.path.join(MODELS_DIR, f"a2c_run{run_id:02d}"))
        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            best_run_id      = run_id
            model.save(os.path.join(MODELS_DIR, "a2c_best"))

        results.append({
            "run": run_id, "learning_rate": hp["learning_rate"],
            "gamma": hp["gamma"], "n_steps": hp["n_steps"],
            "ent_coef": hp["ent_coef"], "vf_coef": hp["vf_coef"],
            "max_grad_norm": hp["max_grad_norm"], "gae_lambda": hp["gae_lambda"],
            "mean_reward": round(mean_r, 3), "std_reward": round(std_r, 3),
        })
        env.close()
        eval_env.close()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "a2c_results.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "a2c_summary.json"), "w") as f:
        json.dump({"best_run": best_run_id,
                   "best_mean_reward": round(best_mean_reward, 3)}, f, indent=2)

    _plot_reward_curves(rewards_per_run,
                        "A2C — Cumulative Reward Curves (All Runs)",
                        os.path.join(PLOTS_DIR, "a2c_reward_curves.png"))
    _plot_entropy_curves(entropy_per_run,
                         "A2C — Entropy Loss per Update (All Runs)",
                         os.path.join(PLOTS_DIR, "a2c_entropy_curves.png"))
    _plot_convergence(df, "A2C",
                      os.path.join(PLOTS_DIR, "a2c_convergence.png"))

    print(f"\n  Best A2C → Run {best_run_id}  "
          f"|  Mean reward: {best_mean_reward:.2f}")
    return df


if __name__ == "__main__":
    train_reinforce()
    train_ppo()
    train_a2c()
    print("\n All policy gradient training complete.")
    print("  Results → models/pg/results/")
    print("  Plots   → models/pg/plots/")