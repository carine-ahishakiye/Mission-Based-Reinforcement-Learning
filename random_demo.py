from environment.custom_env import WildlifeConflictEnv
from environment.rendering import Renderer

ACTION_NAMES = {
    0: "NO ALERT",
    1: "LOW ALERT",
    2: "HIGH ALERT",
    3: "DEPLOY RANGER",
    4: "SEND SMS TO FARMERS",
    5: "ACTIVATE DETERRENT",
}


def run_visual_demo(episodes: int = 3, max_steps: int = 200) -> None:
    env      = WildlifeConflictEnv(render_mode=None)
    renderer = Renderer()
    env.renderer = renderer

    try:
        for ep in range(episodes):
            print(f"\n{'='*55}")
            print(f"  Episode {ep + 1} / {episodes}  [RANDOM ACTIONS - NO MODEL]")
            print(f"{'='*55}")

            obs, info = env.reset()
            ep_reward = 0.0

            for step in range(max_steps):
                action = env.action_space.sample()   # purely random, no model
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                renderer.draw(obs, step + 1, action)

                print(
                    f"  Step {step+1:03d} | "
                    f"Action: {ACTION_NAMES[action]:<22} | "
                    f"Reward: {reward:+.2f} | "
                    f"Dist->Farm: {info['distance_to_farm']:7.1f}m | "
                    f"Season: {info['season'].upper():3} | "
                    f"{'NIGHT' if info['is_night'] else 'DAY  '} | "
                    f"Conflicts: {info['conflict_history']}"
                )

                if terminated:
                    print(f"\n  CONFLICT at step {step+1}! Buffalo reached farmland.")
                    break
                if truncated:
                    reason = (
                        "Safe retreat achieved"
                        if info["distance_to_farm"] >= env.SAFE_RETREAT
                        else "Max steps reached"
                    )
                    print(f"\n  Episode ended - {reason}.")
                    break

            print(f"\n  Total episode reward: {ep_reward:.2f}")

    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    run_visual_demo(episodes=3, max_steps=200)