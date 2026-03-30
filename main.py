import os
import sys
import json
import time
import argparse
import threading
import numpy as np

os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "100,100")
os.environ.setdefault("SDL_VIDEO_CENTERED",   "1")

from http.server import HTTPServer, BaseHTTPRequestHandler

import environment.custom_env
import gymnasium as gym

ACTION_NAMES = {
    0: "NO ALERT",
    1: "LOW ALERT",
    2: "HIGH ALERT",
    3: "DEPLOY RANGER",
    4: "SEND SMS TO FARMERS",
    5: "ACTIVATE DETERRENT",
}

MODEL_PATHS = {
    "dqn": os.path.join("models", "dqn", "dqn_best"),
    "ppo": os.path.join("models", "pg",  "ppo_best"),
    "a2c": os.path.join("models", "pg",  "a2c_best"),
}

SUMMARY_PATHS = {
    "dqn": os.path.join("models", "dqn", "results", "dqn_summary.json"),
    "ppo": os.path.join("models", "pg",  "results", "ppo_summary.json"),
    "a2c": os.path.join("models", "pg",  "results", "a2c_summary.json"),
}

_latest_state: dict = {}


def _build_json_state(obs, action, reward, info, step):
    (step_length, speed, turning_angle,
     dist_farm, ndvi, time_of_day,
     season, dist_water, conflict_hist) = obs
    return {
        "step":              step,
        "action":            int(action),
        "action_name":       ACTION_NAMES[int(action)],
        "reward":            round(float(reward), 3),
        "alert_level":       int(action),
        "distance_to_farm":  round(float(dist_farm),     1),
        "distance_to_water": round(float(dist_water),    1),
        "speed_ms":          round(float(speed),         3),
        "ndvi":              round(float(ndvi),           4),
        "turning_angle":     round(float(turning_angle), 2),
        "step_length":       round(float(step_length),   1),
        "time_of_day":       round(float(time_of_day),   2),
        "season":            "dry" if float(season) == 1 else "wet",
        "is_night":          info.get("is_night",          False),
        "conflict_now":      info.get("conflict",          False),
        "conflict_imminent": info.get("conflict_imminent", False),
        "conflict_history":  int(info.get("conflict_history", 0)),
        "timestamp_unix":    time.time(),
    }


class _APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/predict", "/predict/"):
            payload = json.dumps(_latest_state, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type",   "application/json")
            self.send_header("Content-Length", len(payload))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def _start_api_server(port=5000):
    server = HTTPServer(("0.0.0.0", port), _APIHandler)
    print(f"\n  JSON API live at http://localhost:{port}/predict")
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()


def load_model(algo):
    from stable_baselines3 import DQN, PPO, A2C
    loaders = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
    path = MODEL_PATHS[algo]
    if not os.path.exists(path + ".zip"):
        print(f"[ERROR] No saved model at '{path}.zip'")
        sys.exit(1)
    print(f"  Loading {algo.upper()} from: {path}.zip")
    return loaders[algo].load(path)


def pick_best_algo():
    best_algo, best_reward = None, -np.inf
    for algo, path in SUMMARY_PATHS.items():
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            r = data.get("best_mean_reward", -np.inf)
            if r > best_reward:
                best_reward = r
                best_algo   = algo
    if best_algo is None:
        print("[ERROR] No trained models found.")
        sys.exit(1)
    print(f"  Auto-selected: {best_algo.upper()} (mean reward: {best_reward:.2f})")
    return best_algo


def _pygame_wait(ms, renderer):
    """Wait ms milliseconds while keeping the pygame window alive."""
    import pygame
    clock  = pygame.time.Clock()
    target = pygame.time.get_ticks() + ms
    while pygame.time.get_ticks() < target:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                sys.exit()
        clock.tick(60)


def run_simulation(algo, episodes=3, start_api=False):
    global _latest_state

    print("  ULINZI  Wildlife Conflict Early Warning")
    print("  Live Simulation with Trained Agent")

    if start_api:
        _start_api_server()

    # load model 
    print("\n  Loading model (this takes a moment)...")
    model = load_model(algo)
    print("  Model loaded.\n")

    # creating pygame window and renderer 
    import pygame
    pygame.init()

    from environment.rendering import Renderer
    env      = gym.make("WildlifeConflict-v0", render_mode=None)
    renderer = Renderer()
    env.unwrapped.renderer = renderer

    # Drawing the first frame
    obs, info = env.reset()
    renderer.draw(obs, 0, None)

    try:
        for ep in range(episodes):
            print(f"--- Episode {ep + 1}/{episodes} ---")

            if ep > 0:
                obs, info = env.reset()

            ep_reward = 0.0
            step      = 0
            done      = False

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        renderer.close()
                        env.close()
                        sys.exit()

                action, _ = model.predict(obs, deterministic=True)
                action    = int(action)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step      += 1
                done       = terminated or truncated

                renderer.draw(obs, step, action)
                _pygame_wait(300, renderer) 

                _latest_state = _build_json_state(obs, action, reward, info, step)

                print(
                    f"  Step {step:03d} | "
                    f"Action: {ACTION_NAMES[action]:<22} | "
                    f"Reward: {reward:+.2f} | "
                    f"Dist->Farm: {info['distance_to_farm']:7.1f}m | "
                    f"Season: {info['season'].upper():3} | "
                    f"{'NIGHT' if info['is_night'] else 'DAY  '} | "
                    f"Conflicts: {info['conflict_history']}"
                )

                if terminated:
                    print(f"\n  CONFLICT at step {step}! Buffalo reached farmland.")
                elif truncated:
                    reason = (
                        "Safe retreat achieved"
                        if info["distance_to_farm"] >= env.unwrapped.SAFE_RETREAT
                        else "Max steps reached"
                    )
                    print(f"\n  {reason} at step {step}.")

            print(f"  Episode {ep + 1} total reward: {ep_reward:.2f}\n")
            _pygame_wait(1500, renderer)

    finally:
        renderer.close()
        env.close()

    print("  Simulation complete.")


def main():
    parser = argparse.ArgumentParser(description="ULINZI - run best RL agent.")
    parser.add_argument("--algo",     type=str,
                        choices=["dqn", "ppo", "a2c", "auto"], default="auto")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--api",      action="store_true")
    args = parser.parse_args()
    algo = pick_best_algo() if args.algo == "auto" else args.algo
    run_simulation(algo=algo, episodes=args.episodes, start_api=args.api)


if __name__ == "__main__":
    main()