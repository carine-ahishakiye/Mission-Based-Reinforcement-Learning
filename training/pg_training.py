import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import UlinziEnv

os.makedirs("models/pg", exist_ok=True)
os.makedirs("logs/pg", exist_ok=True)


def make_env(seed=0):
    env = UlinziEnv(max_steps=48)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


class EntropyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.entropies = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        if hasattr(self.model, "policy") and hasattr(self.model.logger, "name_to_value"):
            ent = self.model.logger.name_to_value.get("train/entropy_loss", None)
            if ent is not None:
                self.entropies.append(float(ent))
        return True


TIMESTEPS = 80000
EVAL_EPISODES = 20


PPO_RUNS = [
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 128,  "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.01,  "clip_range": 0.2,  "gae_lambda": 0.95},
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 256,  "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.01,  "clip_range": 0.2,  "gae_lambda": 0.95},
    {"learning_rate": 5e-4, "gamma": 0.95, "n_steps": 128,  "batch_size": 32,  "n_epochs": 5,  "ent_coef": 0.02,  "clip_range": 0.3,  "gae_lambda": 0.9},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512,  "batch_size": 128, "n_epochs": 10, "ent_coef": 0.005, "clip_range": 0.2,  "gae_lambda": 0.95},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 128,  "batch_size": 64,  "n_epochs": 15, "ent_coef": 0.0,   "clip_range": 0.15, "gae_lambda": 0.98},
    {"learning_rate": 2e-4, "gamma": 0.97, "n_steps": 256,  "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.01,  "clip_range": 0.25, "gae_lambda": 0.92},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 128,  "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.05,  "clip_range": 0.2,  "gae_lambda": 0.95},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 256,  "batch_size": 128, "n_epochs": 8,  "ent_coef": 0.01,  "clip_range": 0.2,  "gae_lambda": 0.95},
    {"learning_rate": 1e-4, "gamma": 0.98, "n_steps": 512,  "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.02,  "clip_range": 0.3,  "gae_lambda": 0.97},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 256, "n_epochs": 10, "ent_coef": 0.01,  "clip_range": 0.2,  "gae_lambda": 0.95},
]

A2C_RUNS = [
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 1.0},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 1.0},
    {"learning_rate": 5e-4, "gamma": 0.95, "n_steps": 10, "ent_coef": 0.02,  "vf_coef": 0.25, "max_grad_norm": 0.5, "gae_lambda": 0.9},
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 20, "ent_coef": 0.005, "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 0.95},
    {"learning_rate": 2e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.01,  "vf_coef": 1.0,  "max_grad_norm": 1.0, "gae_lambda": 1.0},
    {"learning_rate": 1e-3, "gamma": 0.97, "n_steps": 8,  "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 0.95},
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.05,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 1.0},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 16, "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 0.97},
    {"learning_rate": 5e-4, "gamma": 0.98, "n_steps": 5,  "ent_coef": 0.02,  "vf_coef": 0.75, "max_grad_norm": 0.5, "gae_lambda": 1.0},
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 32, "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 0.95},
]

REINFORCE_RUNS = [
    {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64,  "entropy_coef": 0.01, "baseline": True},
    {"learning_rate": 5e-4, "gamma": 0.99, "hidden_size": 128, "entropy_coef": 0.01, "baseline": True},
    {"learning_rate": 1e-3, "gamma": 0.95, "hidden_size": 64,  "entropy_coef": 0.0,  "baseline": False},
    {"learning_rate": 2e-3, "gamma": 0.99, "hidden_size": 64,  "entropy_coef": 0.05, "baseline": True},
    {"learning_rate": 5e-4, "gamma": 0.97, "hidden_size": 128, "entropy_coef": 0.02, "baseline": True},
    {"learning_rate": 1e-4, "gamma": 0.99, "hidden_size": 256, "entropy_coef": 0.01, "baseline": True},
    {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64,  "entropy_coef": 0.1,  "baseline": False},
    {"learning_rate": 3e-4, "gamma": 0.98, "hidden_size": 128, "entropy_coef": 0.01, "baseline": True},
    {"learning_rate": 2e-3, "gamma": 0.95, "hidden_size": 64,  "entropy_coef": 0.0,  "baseline": True},
    {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64,  "entropy_coef": 0.02, "baseline": True},
]


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_reinforce(run_idx, params, timesteps=80000):
    print(f"\n  REINFORCE Run {run_idx + 1}/10 | LR={params['learning_rate']} | gamma={params['gamma']} | hidden={params['hidden_size']}")

    env = UlinziEnv(max_steps=48)
    env.reset(seed=run_idx)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, act_dim, params["hidden_size"])
    for m in policy.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)
    policy_opt = optim.Adam(policy.parameters(), lr=params["learning_rate"])

    value_net = None
    value_opt = None
    if params["baseline"]:
        value_net = ValueNetwork(obs_dim, params["hidden_size"])
        for m in value_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        value_opt = optim.Adam(value_net.parameters(), lr=params["learning_rate"])

    episode_rewards = []
    entropies = []
    total_steps = 0

    while total_steps < timesteps:
        obs, _ = env.reset()
        states, rewards, log_probs_list, ep_entropies = [], [], [], []
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                logits = policy.net(obs_t)
                logits = logits - logits.max(dim=-1, keepdim=True).values
                probs = torch.softmax(logits, dim=-1).clamp(min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            action = Categorical(probs).sample().item()

            logits_g = policy.net(obs_t)
            logits_g = logits_g - logits_g.max(dim=-1, keepdim=True).values.detach()
            probs_g = torch.softmax(logits_g, dim=-1).clamp(min=1e-8)
            probs_g = probs_g / probs_g.sum(dim=-1, keepdim=True)
            dist = Categorical(probs_g)
            log_prob = dist.log_prob(torch.tensor(action))
            ep_entropies.append(dist.entropy().item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(obs)
            rewards.append(reward)
            log_probs_list.append(log_prob)
            obs = next_obs
            total_steps += 1

        episode_rewards.append(sum(rewards))
        entropies.extend(ep_entropies)

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + params["gamma"] * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        if value_net is not None:
            states_t = torch.FloatTensor(np.array(states))
            values = value_net(states_t).squeeze(-1)
            advantages = returns_t - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            value_loss = nn.MSELoss()(values, returns_t)
            value_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
            value_opt.step()
        else:
            advantages = returns_t

        log_probs_t = torch.stack(log_probs_list)
        entropy_bonus = torch.tensor(ep_entropies).mean()
        policy_loss = -(log_probs_t * advantages.detach()).mean() - params["entropy_coef"] * entropy_bonus

        policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        policy_opt.step()

    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits = policy.net(obs_t)
                logits = logits - logits.max(dim=-1, keepdim=True).values
                probs = torch.softmax(logits, dim=-1)
            action = probs.argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_r += reward
            done = terminated or truncated
        eval_rewards.append(ep_r)

    env.close()
    mean_r = float(np.mean(eval_rewards))
    std_r = float(np.std(eval_rewards))
    print(f"  Mean Reward: {mean_r:.3f} ± {std_r:.3f}")

    return policy, {
        "run": run_idx + 1,
        "mean_reward": round(mean_r, 3),
        "std_reward": round(std_r, 3),
        "episode_rewards": episode_rewards,
        "entropies": entropies,
        **params,
    }


def train_sb3_run(algo_class, algo_name, run_idx, params):
    print(f"\n  {algo_name} Run {run_idx + 1}/10 | LR={params['learning_rate']} | gamma={params['gamma']}")

    env = make_env(seed=run_idx)
    eval_env = make_env(seed=100 + run_idx)

    callback = EntropyCallback()

    model = algo_class("MlpPolicy", env, verbose=0, seed=run_idx, **params)
    model.learn(total_timesteps=TIMESTEPS, callback=callback, progress_bar=False)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)

    env.close()
    eval_env.close()

    result = {
        "run": run_idx + 1,
        "mean_reward": round(float(mean_reward), 3),
        "std_reward": round(float(std_reward), 3),
        "episode_rewards": callback.episode_rewards,
        "entropies": callback.entropies,
        **params,
    }
    print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    return model, result


def plot_pg_results(ppo_results, a2c_results, reinforce_results):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Policy Gradient Methods — ULINZI Hyperparameter Experiments", fontsize=14, fontweight="bold")

    algo_data = [
        ("PPO", ppo_results, axes[0]),
        ("A2C", a2c_results, axes[1]),
        ("REINFORCE", reinforce_results, axes[2]),
    ]

    for algo_name, results, row_axes in algo_data:
        ax = row_axes[0]
        for r in results:
            ep_r = r.get("episode_rewards", [])
            if len(ep_r) >= 10:
                smoothed = np.convolve(ep_r, np.ones(10) / 10, mode="valid")
            else:
                smoothed = ep_r
            if len(smoothed) > 0:
                ax.plot(smoothed, alpha=0.6, label=f"R{r['run']}")
        ax.set_title(f"{algo_name} — Episode Rewards (Smoothed)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = row_axes[1]
        runs = [r["run"] for r in results]
        means = [r["mean_reward"] for r in results]
        stds = [r["std_reward"] for r in results]
        bars = ax.bar(runs, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        best_idx = int(np.argmax(means))
        bars[best_idx].set_color("gold")
        ax.set_title(f"{algo_name} — Eval Mean Reward per Run")
        ax.set_xlabel("Run")
        ax.set_ylabel("Mean Reward")
        ax.set_xticks(runs)
        ax.grid(True, alpha=0.3, axis="y")

        ax = row_axes[2]
        for r in results:
            ents = r.get("entropies", [])
            if len(ents) >= 20:
                smoothed = np.convolve(ents, np.ones(20) / 20, mode="valid")
            else:
                smoothed = ents
            if len(smoothed) > 0:
                ax.plot(smoothed, alpha=0.6, label=f"R{r['run']}")
        ax.set_title(f"{algo_name} — Entropy over Training")
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "models/pg/pg_hyperparameter_results.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {path}")


def plot_convergence(ppo_results, a2c_results, reinforce_results):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Convergence Comparison — Best Run per Algorithm", fontsize=13, fontweight="bold")

    for name, results, color in [("PPO", ppo_results, "blue"), ("A2C", a2c_results, "orange"), ("REINFORCE", reinforce_results, "green")]:
        best = max(results, key=lambda r: r["mean_reward"])
        ep_r = best.get("episode_rewards", [])
        if len(ep_r) >= 15:
            smoothed = np.convolve(ep_r, np.ones(15) / 15, mode="valid")
            ax.plot(smoothed, label=f"{name} (Run {best['run']}, Mean={best['mean_reward']:.2f})", color=color, linewidth=2)
        else:
            ax.axhline(best["mean_reward"], label=f"{name} (Run {best['run']}, Mean={best['mean_reward']:.2f})", color=color, linewidth=2, linestyle="--")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward (Smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "models/pg/convergence_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Convergence plot saved: {path}")


def main():
    print("\n" + "=" * 60)
    print("  ULINZI — REINFORCE Training (PPO + A2C loaded from disk)")
    print("=" * 60)

    ppo_path = "models/pg/ppo_results_summary.json"
    a2c_path = "models/pg/a2c_results_summary.json"

    if not os.path.exists(ppo_path) or not os.path.exists(a2c_path):
        print("\n  ERROR: PPO/A2C result files not found.")
        print(f"  Expected: {ppo_path}")
        print(f"  Expected: {a2c_path}")
        sys.exit(1)

    with open(ppo_path) as f:
        ppo_results = json.load(f)
    with open(a2c_path) as f:
        a2c_results = json.load(f)

    best_ppo_reward = max(r["mean_reward"] for r in ppo_results)
    best_a2c_reward = max(r["mean_reward"] for r in a2c_results)
    best_ppo_run = max(ppo_results, key=lambda r: r["mean_reward"])["run"]
    best_a2c_run = max(a2c_results, key=lambda r: r["mean_reward"])["run"]

    print(f"\n  Loaded PPO  — Best: {best_ppo_reward:.3f} (Run {best_ppo_run})")
    print(f"  Loaded A2C  — Best: {best_a2c_reward:.3f} (Run {best_a2c_run})")

    print("\n[REINFORCE] Training 10 runs...")
    reinforce_results = []
    best_reinforce_model = None
    best_reinforce_reward = -np.inf

    for i, params in enumerate(REINFORCE_RUNS):
        model, result = train_reinforce(i, params)
        reinforce_results.append(result)
        if result["mean_reward"] > best_reinforce_reward:
            best_reinforce_reward = result["mean_reward"]
            best_reinforce_model = model

    torch.save(best_reinforce_model.state_dict(), "models/pg/reinforce_best.pt")
    print(f"\n  Best REINFORCE saved — Mean Reward: {best_reinforce_reward:.3f}")

    def clean(results):
        return [{k: v for k, v in r.items() if k not in ("episode_rewards", "entropies")} for r in results]

    with open("models/pg/reinforce_results_summary.json", "w") as f:
        json.dump(clean(reinforce_results), f, indent=2)

    plot_pg_results(ppo_results, a2c_results, reinforce_results)
    plot_convergence(ppo_results, a2c_results, reinforce_results)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    reinforce_best_run = max(reinforce_results, key=lambda r: r["mean_reward"])["run"]
    for algo, reward, note in [
        ("PPO",       best_ppo_reward,       f"Run {best_ppo_run}"),
        ("A2C",       best_a2c_reward,       f"Run {best_a2c_run}"),
        ("REINFORCE", best_reinforce_reward, f"Run {reinforce_best_run}"),
    ]:
        print(f"  {algo:<12} | {reward:>10.3f} | {note}")


if __name__ == "__main__":
    main()