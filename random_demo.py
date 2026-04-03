import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import UlinziEnv
from environment.rendering import UlinziRenderer


def main():
    env = UlinziEnv(render_mode="human", max_steps=48)
    renderer = UlinziRenderer()
    env.renderer = renderer

    print("\n" + "=" * 60)
    print("  ULINZI — Random Agent Demo")
    print("  No model loaded, actions are sampled uniformly at random")
    print("=" * 60)

    for ep in range(3):
        obs, info = env.reset(seed=ep)
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

        print(f"\n  Episode {ep + 1} done | Total Reward: {ep_reward:.2f} | "
              f"Successful Alerts: {info['successful_alerts']} | False Alarms: {info['false_alarms']}")

    env.close()


if __name__ == "__main__":
    main()