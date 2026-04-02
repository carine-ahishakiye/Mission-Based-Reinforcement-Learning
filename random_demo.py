import sys
import pygame

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

STEP_DELAY_MS    = 300   
EPISODE_PAUSE_MS = 1500  


def _pump_events(renderer: Renderer, wait_ms: int) -> None:
    """
    Wait `wait_ms` milliseconds while keeping the pygame window responsive.
    Exits the whole program cleanly if the user clicks the window's X button.
    """
    clock  = pygame.time.Clock()
    target = pygame.time.get_ticks() + wait_ms
    while pygame.time.get_ticks() < target:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                sys.exit()
        clock.tick(60)


def run_visual_demo(episodes: int = 3, max_steps: int = 200) -> None:
    env      = WildlifeConflictEnv(render_mode=None)
    renderer = Renderer()

    try:
        for ep in range(episodes):
            print(f"\n{'='*55}")
            print(f"  Episode {ep + 1} / {episodes}  [RANDOM ACTIONS - NO MODEL]")
            print(f"{'='*55}")

            obs, info = env.reset()
            ep_reward = 0.0

            # Drawing the initial state before the first step
            renderer.draw(obs, 0, None)
            _pump_events(renderer, STEP_DELAY_MS)

            for step in range(max_steps):

               
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        renderer.close()
                        env.close()
                        return

                action = env.action_space.sample()       
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                renderer.draw(obs, step + 1, action)
                _pump_events(renderer, STEP_DELAY_MS)       

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
                    print(f"\n  Episode ended — {reason}.")
                    break

            print(f"\n  Total episode reward: {ep_reward:.2f}")

            # Pause between episodes so the final frame is visible
            _pump_events(renderer, EPISODE_PAUSE_MS)

    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    run_visual_demo(episodes=3, max_steps=200)