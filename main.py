import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import UlinziEnv
from environment.rendering import UlinziRenderer


def load_best_model():
    candidates = [
        ("DQN",       "models/dqn/dqn_best.zip",   "dqn"),
        ("PPO",       "models/pg/ppo_best.zip",     "ppo"),
        ("A2C",       "models/pg/a2c_best.zip",     "a2c"),
        ("REINFORCE", "models/pg/reinforce_best.pt","reinforce"),
    ]

    score_files = {
        "dqn":       "models/dqn/dqn_results_summary.json",
        "ppo":       "models/pg/ppo_results_summary.json",
        "a2c":       "models/pg/a2c_results_summary.json",
        "reinforce": "models/pg/reinforce_results_summary.json",
    }

    scores = {}
    for key, path in score_files.items():
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            best = max(data, key=lambda r: r["mean_reward"])
            scores[key] = best["mean_reward"]

    best_algo = None
    best_score = -np.inf
    for name, model_path, key in candidates:
        if os.path.exists(model_path) and key in scores:
            if scores[key] > best_score:
                best_score = scores[key]
                best_algo = (name, model_path, key)

    if best_algo is None:
        for name, model_path, key in candidates:
            if os.path.exists(model_path):
                best_algo = (name, model_path, key)
                break

    return best_algo, scores


def load_model_object(algo_name, model_path, algo_key):
    if algo_key == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif algo_key == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif algo_key == "a2c":
        from stable_baselines3 import A2C
        return A2C.load(model_path)
    elif algo_key == "reinforce":
        import torch
        import torch.nn as nn

        class PolicyNetwork(nn.Module):
            def __init__(self, obs_dim, act_dim, hidden_size):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, act_dim),
                )
            def forward(self, x):
                return torch.softmax(self.net(x), dim=-1)

        model = PolicyNetwork(9, 6, 64)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model


def get_action(model, algo_key, obs):
    if algo_key == "reinforce":
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = model.net(obs_t)
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=1).item()
    else:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)


def run_with_model(model, algo_name, algo_key, env, renderer, n_episodes=3, seed=0):
    print(f"\n{'='*60}")
    print(f"  Running best agent: {algo_name}")
    print(f"  Episodes: {n_episodes}")
    print(f"{'='*60}")

    total_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        renderer.reset(obs)
        done = False
        ep_reward = 0.0
        step = 0
        last_info = info

        print(f"\n  Episode {ep + 1}/{n_episodes} | Crossing: {info['crossing_episode']} | TTC: {info['time_to_crossing']}")
        print(f"  {'Step':>4} | {'Action':<24} | {'Risk':>6} | {'Reward':>8} | {'Cumulative':>10} | Event")
        print(f"  {'-'*80}")

        while not done:
            action = get_action(model, algo_key, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1
            last_info = info

            renderer.render(obs, action, reward, info)

            print(
                f"  {step:>4} | {info['action_label']:<24} | {info['risk_score']:>6.3f} | "
                f"{reward:>+8.2f} | {ep_reward:>+10.2f} | {info.get('event', '')}"
            )

        total_rewards.append(ep_reward)
        print(f"\n  Episode {ep + 1} complete — Total Reward: {ep_reward:.2f} | Successful Alerts: {last_info['successful_alerts']} | False Alarms: {last_info['false_alarms']}")

    print(f"\n{'='*60}")
    print(f"  Run complete — Mean Episode Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"{'='*60}")


def run_random_demo(env, renderer, n_episodes=2, seed=0):
    print(f"\n{'='*60}")
    print("  Running RANDOM AGENT demo (no model)")
    print(f"{'='*60}")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        renderer.reset(obs)
        done = False
        ep_reward = 0.0
        step = 0

        print(f"\n  Episode {ep + 1} | Crossing: {info['crossing_episode']} | TTC: {info['time_to_crossing']}")
        print(f"  {'Step':>4} | {'Action':<24} | {'Risk':>6} | {'Reward':>8} | {'Cumulative':>10} | Event")
        print(f"  {'-'*80}")

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1
            renderer.render(obs, action, reward, info)
            print(
                f"  {step:>4} | {info['action_label']:<24} | {info['risk_score']:>6.3f} | "
                f"{reward:>+8.2f} | {ep_reward:>+10.2f} | {info.get('event', '')}"
            )

        print(f"\n  Episode {ep + 1} done — Total Reward: {ep_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description="ULINZI — Wildlife Alert RL Agent")
    parser.add_argument("--algo",     type=str, default=None, help="Force algorithm: dqn, ppo, a2c, reinforce")
    parser.add_argument("--episodes", type=int, default=3,    help="Number of evaluation episodes")
    parser.add_argument("--seed",     type=int, default=42,   help="Random seed")
    parser.add_argument("--random",   action="store_true",    help="Run random agent demo instead")
    parser.add_argument("--headless", action="store_true",    help="Disable pygame rendering")
    args = parser.parse_args()

    render_mode = None if args.headless else "human"

    env = UlinziEnv(render_mode=render_mode, max_steps=48)

    if render_mode == "human":
        renderer = UlinziRenderer()
        env.renderer = renderer
    else:
        class NoOpRenderer:
            def reset(self, obs): pass
            def render(self, obs, action, reward, info): pass
        renderer = NoOpRenderer()

    if args.random:
        run_random_demo(env, renderer, n_episodes=args.episodes, seed=args.seed)
        env.close()
        return

    best_algo, scores = load_best_model()

    if args.algo:
        override_map = {
            "dqn":       ("DQN",       "models/dqn/dqn_best.zip",    "dqn"),
            "ppo":       ("PPO",       "models/pg/ppo_best.zip",      "ppo"),
            "a2c":       ("A2C",       "models/pg/a2c_best.zip",      "a2c"),
            "reinforce": ("REINFORCE", "models/pg/reinforce_best.pt", "reinforce"),
        }
        if args.algo.lower() in override_map:
            best_algo = override_map[args.algo.lower()]

    if best_algo is None:
        print("  No trained models found. Running random agent demo.")
        run_random_demo(env, renderer, n_episodes=args.episodes, seed=args.seed)
        env.close()
        return

    algo_name, model_path, algo_key = best_algo

    print(f"\n  Loading {algo_name} from {model_path}")
    if scores:
        print("  Evaluation scores from training:")
        for k, v in scores.items():
            marker = " <-- best" if v == max(scores.values()) else ""
            print(f"    {k.upper():<12}: {v:>8.3f}{marker}")

    model = load_model_object(algo_name, model_path, algo_key)
    run_with_model(model, algo_name, algo_key, env, renderer, n_episodes=args.episodes, seed=args.seed)
    env.close()


if __name__ == "__main__":
    main()