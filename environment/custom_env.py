import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


ALERT_LABELS = [
    "NO_ALERT",
    "COMMUNITY_SMS",
    "RANGER_DISPATCH",
    "SCARE_DEVICE_ACTIVATE",
    "ELEVATED_WATCH",
    "EMERGENCY_BROADCAST",
]

OBS_LOW = np.array([
    0.0,
    0.0,
    0.0,
    -1.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    0.0,
], dtype=np.float32)

OBS_HIGH = np.array([
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
], dtype=np.float32)

FEATURE_NAMES = [
    "boundary_proximity",
    "speed_norm",
    "heading_change_norm",
    "displacement_x_norm",
    "displacement_y_norm",
    "time_of_day_norm",
    "vegetation_density",
    "herd_cohesion",
    "recent_crossing_history",
]


class UlinziEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 48):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=OBS_LOW,
            high=OBS_HIGH,
            dtype=np.float32,
        )

        self._step = 0
        self._state: np.ndarray = np.zeros(9, dtype=np.float32)
        self._crossed = False
        self._alert_history: list = []
        self._crossing_imminent = False
        self._time_to_crossing: int = 0
        self._total_reward: float = 0.0
        self._false_alarms: int = 0
        self._successful_alerts: int = 0

        self.renderer = None

    def _sample_episode_scenario(self) -> Tuple[bool, int]:
        crossing_episode = self.np_random.random() < 0.6
        if crossing_episode:
            time_to_crossing = int(self.np_random.integers(4, self.max_steps - 4))
        else:
            time_to_crossing = self.max_steps + 1
        return crossing_episode, time_to_crossing

    def _build_observation(self) -> np.ndarray:
        t = self._step
        ttc = self._time_to_crossing

        if ttc <= self.max_steps:
            progress = t / max(ttc, 1)
            boundary_proximity = float(np.clip(0.2 + 0.75 * progress + self.np_random.normal(0, 0.04), 0, 1))
            speed_norm = float(np.clip(0.15 + 0.6 * progress + self.np_random.normal(0, 0.05), 0, 1))
            heading_change = float(np.clip(0.3 - 0.2 * progress + self.np_random.normal(0, 0.06), 0, 1))
            dx = float(np.clip(0.05 + 0.5 * progress * np.cos(t * 0.3) + self.np_random.normal(0, 0.04), -1, 1))
            dy = float(np.clip(0.05 + 0.5 * progress * np.sin(t * 0.2) + self.np_random.normal(0, 0.04), -1, 1))
        else:
            boundary_proximity = float(np.clip(self.np_random.uniform(0.05, 0.55) + self.np_random.normal(0, 0.04), 0, 1))
            speed_norm = float(np.clip(self.np_random.uniform(0.05, 0.45) + self.np_random.normal(0, 0.04), 0, 1))
            heading_change = float(np.clip(self.np_random.uniform(0.3, 0.9) + self.np_random.normal(0, 0.05), 0, 1))
            dx = float(np.clip(self.np_random.uniform(-0.3, 0.3), -1, 1))
            dy = float(np.clip(self.np_random.uniform(-0.3, 0.3), -1, 1))

        hour = (t * 0.5) % 24
        time_norm = float(np.sin(np.pi * hour / 24))

        vegetation = float(np.clip(self.np_random.uniform(0.2, 0.9) + self.np_random.normal(0, 0.03), 0, 1))
        cohesion = float(np.clip(self.np_random.uniform(0.3, 1.0) + self.np_random.normal(0, 0.04), 0, 1))
        crossing_history = float(min(sum(1 for a in self._alert_history[-6:] if a >= 2) / 6.0, 1.0))

        obs = np.array([
            boundary_proximity,
            speed_norm,
            heading_change,
            dx,
            dy,
            time_norm,
            vegetation,
            cohesion,
            crossing_history,
        ], dtype=np.float32)

        self._state = obs
        return obs

    def _compute_reward(self, action: int) -> Tuple[float, Dict[str, Any]]:
        bp = float(self._state[0])
        spd = float(self._state[1])
        risk_score = 0.5 * bp + 0.3 * spd + 0.2 * float(self._state[7])

        steps_remaining = self._time_to_crossing - self._step
        crossing_imminent = 0 < steps_remaining <= 6

        reward = 0.0
        info_reward = {}

        if crossing_imminent:
            if action == 0:
                reward = -8.0
                info_reward["event"] = "missed_critical_alert"
            elif action == 1:
                reward = 2.0
                info_reward["event"] = "community_alerted"
                self._successful_alerts += 1
            elif action == 2:
                reward = 4.0
                info_reward["event"] = "rangers_dispatched"
                self._successful_alerts += 1
            elif action == 3:
                reward = 3.5
                info_reward["event"] = "scare_device_activated"
                self._successful_alerts += 1
            elif action == 4:
                reward = 1.5
                info_reward["event"] = "watch_elevated"
                self._successful_alerts += 1
            elif action == 5:
                reward = 5.0
                info_reward["event"] = "emergency_broadcast"
                self._successful_alerts += 1

        elif risk_score > 0.65:
            if action == 0:
                reward = -3.0
                info_reward["event"] = "ignored_high_risk"
            elif action in [1, 4]:
                reward = 2.5
                info_reward["event"] = "appropriate_precaution"
            elif action in [2, 3]:
                reward = 1.5
                info_reward["event"] = "overreaction_moderate_risk"
            elif action == 5:
                reward = -1.5
                info_reward["event"] = "emergency_overuse"
                self._false_alarms += 1

        elif risk_score < 0.3:
            if action == 0:
                reward = 1.0
                info_reward["event"] = "correct_silence"
            elif action == 5:
                reward = -4.0
                info_reward["event"] = "false_emergency"
                self._false_alarms += 1
            elif action in [2, 3]:
                reward = -2.0
                info_reward["event"] = "unnecessary_dispatch"
                self._false_alarms += 1
            elif action in [1, 4]:
                reward = -0.5
                info_reward["event"] = "minor_false_alarm"
                self._false_alarms += 1

        else:
            if action == 4:
                reward = 1.0
                info_reward["event"] = "watch_maintained"
            elif action == 0:
                reward = 0.0
                info_reward["event"] = "passive_observation"
            elif action == 1:
                reward = 0.5
                info_reward["event"] = "community_informed"
            elif action in [2, 3, 5]:
                reward = -0.5
                info_reward["event"] = "premature_escalation"

        if steps_remaining == self._time_to_crossing and self._time_to_crossing <= self.max_steps:
            self._crossed = True

        return reward, info_reward

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._step = 0
        self._crossed = False
        self._alert_history = []
        self._false_alarms = 0
        self._successful_alerts = 0
        self._total_reward = 0.0

        self._crossing_episode, self._time_to_crossing = self._sample_episode_scenario()
        obs = self._build_observation()

        info = {
            "crossing_episode": self._crossing_episode,
            "time_to_crossing": self._time_to_crossing,
            "step": self._step,
        }

        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.reset(obs)

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._alert_history.append(action)
        reward, reward_info = self._compute_reward(action)
        self._total_reward += reward
        self._step += 1

        obs = self._build_observation()

        terminated = self._crossed or self._step >= self.max_steps
        truncated = False

        info = {
            "step": self._step,
            "action_label": ALERT_LABELS[action],
            "risk_score": float(0.5 * self._state[0] + 0.3 * self._state[1] + 0.2 * self._state[7]),
            "boundary_proximity": float(self._state[0]),
            "false_alarms": self._false_alarms,
            "successful_alerts": self._successful_alerts,
            "total_reward": self._total_reward,
            "crossing_imminent": 0 < (self._time_to_crossing - self._step) <= 6,
            **reward_info,
        }

        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.render(obs, action, reward, info)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_rgb_array(self) -> np.ndarray:
        try:
            import pygame
            import pygame.surfarray as surfarray
            if not hasattr(self, "_rgb_surface"):
                pygame.init()
                self._rgb_surface = pygame.Surface((900, 600))
            from environment.rendering import UlinziRenderer
            renderer = UlinziRenderer(surface=self._rgb_surface)
            renderer.draw_frame(self._state, self._alert_history[-1] if self._alert_history else 0, 0.0, {})
            return np.transpose(surfarray.array3d(self._rgb_surface), (1, 0, 2))
        except Exception:
            return np.zeros((600, 900, 3), dtype=np.uint8)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "features": {
                FEATURE_NAMES[i]: float(self._state[i]) for i in range(9)
            },
            "step": self._step,
            "total_reward": self._total_reward,
            "false_alarms": self._false_alarms,
            "successful_alerts": self._successful_alerts,
            "alert_history": [ALERT_LABELS[a] for a in self._alert_history],
        }