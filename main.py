from environment.custom_env import WildlifeConflictEnv
from environment.rendering import Renderer
import numpy as np

def run_visual_demo(episodes=1, max_steps=200):
    env = WildlifeConflictEnv(render_mode="human")
    env.renderer = Renderer()

    for ep in range(episodes):
        print(f"\n Episode {ep+1} ")
        obs, info = env.reset()

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            env.renderer.draw(obs, step, action)

            print(f"Action: {action}, Reward: {reward}, Info: {info}")

        print("Episode finished.")

    env.renderer.close()

if __name__ == "__main__":
    run_visual_demo()
