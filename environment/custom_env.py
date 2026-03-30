import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import registry, register


if "WildlifeConflict-v0" not in registry:
    register(
        id="WildlifeConflict-v0",
        entry_point="environment.custom_env:WildlifeConflictEnv",
    )


class WildlifeConflictEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}
    NO_ALERT           = 0
    LOW_ALERT          = 1
    HIGH_ALERT         = 2
    DEPLOY_RANGER      = 3
    SEND_SMS           = 4
    ACTIVATE_DETERRENT = 5

    # Distance thresholds 
    CONFLICT_DISTANCE  = 200
    IMMINENT_DISTANCE  = 1000
    APPROACH_DISTANCE  = 3000
    SAFE_RETREAT       = 9000

    MAX_STEPS = 200

    #  Constructor 
    def __init__(self, render_mode=None):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"Invalid render_mode '{render_mode}'. "
            f"Choose from {self.metadata['render_modes']}"
        )
        self.render_mode = render_mode

        # Observation space bounds
        low = np.array(
            [0.0,   0.0,   0.0,   0.0,  -1.0,  0.0,  0.0,  0.0,  0.0],
            dtype=np.float32,
        )
        high = np.array(
            [5000.0, 10.0, 180.0, 10000.0, 1.0, 23.0, 1.0, 10000.0, 10.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(6)

        # Internal state
        self.state            = None
        self.current_step     = 0
        self.season           = 0
        self.conflict_history = 0
        self.renderer = None

    #  Reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step     = 0
        self.conflict_history = 0
        self.season           = int(self.np_random.integers(0, 2))

        time_of_day = float(self.np_random.integers(0, 24))

      
        if self.season == 1:
            dist_water = self.np_random.uniform(3000, 8000)
        else:
            dist_water = self.np_random.uniform(500, 3000)

        self.state = np.array(
            [
                self.np_random.uniform(50,   300),    
                self.np_random.uniform(0.1,  1.0),  
                self.np_random.uniform(60,   180),   
                self.np_random.uniform(5000, 8500),  
                self.np_random.uniform(0.3,  1.0),    
                time_of_day,                          
                float(self.season),                   
                dist_water,                          
                0.0,                                  
            ],
            dtype=np.float32,
        )

        return self.state, {}

    def step(self, action):
        (step_length, speed, turning_angle,
         distance_to_farmland, ndvi, time_of_day,
         _, distance_to_water, _conflict_hist) = self.state

        season = self.season
        # Season effect
        if season == 1:  # dry
            step_length += self.np_random.uniform(50,  250)
            speed       += self.np_random.uniform(0.1, 0.8)
        else:            # wet
            step_length += self.np_random.uniform(10,  80)
            speed       += self.np_random.uniform(0.0, 0.3)

        # Night effect
        is_night = (time_of_day >= 18) or (time_of_day <= 6)
        if is_night:
            speed       += self.np_random.uniform(0.1, 0.4)
            step_length += self.np_random.uniform(20,  100)

        # Turning angle narrows as animal approaches farmland
        if distance_to_farmland < self.APPROACH_DISTANCE:
            turning_angle = max(5.0, turning_angle - self.np_random.uniform(5, 20))
        else:
            turning_angle = min(180.0, turning_angle + self.np_random.uniform(0, 10))

        # NDVI dynamics
        if season == 1:
            ndvi = max(-1.0, ndvi - self.np_random.uniform(0.01, 0.05))
        else:
            ndvi = min(1.0,  ndvi + self.np_random.uniform(0.0,  0.02))

        # Water distance dynamics
        if season == 1:
            distance_to_water = max(0.0, distance_to_water - self.np_random.uniform(50, 200))
        else:
            distance_to_water = min(10000.0, distance_to_water + self.np_random.uniform(0, 50))

        # Buffalo moves toward farmland; deterrent halves effective movement
        deterrent_factor     = 0.5 if action == self.ACTIVATE_DETERRENT else 1.0
        move_toward_farm     = speed * self.np_random.uniform(50, 150) * deterrent_factor
        distance_to_farmland = max(0.0, distance_to_farmland - move_toward_farm)

        # Advance time by 30 minutes per step
        time_of_day = (time_of_day + 0.5) % 24

        # Clip all values to valid observation bounds
        step_length          = np.clip(step_length,          0.0,  5000.0)
        speed                = np.clip(speed,                0.0,    10.0)
        turning_angle        = np.clip(turning_angle,        0.0,   180.0)
        distance_to_farmland = np.clip(distance_to_farmland, 0.0, 10000.0)
        ndvi                 = np.clip(ndvi,                -1.0,     1.0)
        distance_to_water    = np.clip(distance_to_water,   0.0, 10000.0)

        # Conflict flags
        conflict_now      = distance_to_farmland <= self.CONFLICT_DISTANCE
        conflict_imminent = distance_to_farmland <= self.IMMINENT_DISTANCE

        if conflict_now:
            self.conflict_history = min(10, self.conflict_history + 1)

        # Update state
        self.state = np.array(
            [
                step_length, speed, turning_angle,
                distance_to_farmland, ndvi, time_of_day,
                float(self.season),
                distance_to_water,
                float(self.conflict_history),
            ],
            dtype=np.float32,
        )

        reward = self._calculate_reward(
            action, conflict_now, conflict_imminent, distance_to_farmland
        )

        self.current_step += 1

        terminated  = bool(conflict_now)
        safe_retreat = distance_to_farmland >= self.SAFE_RETREAT
        truncated   = bool(self.current_step >= self.MAX_STEPS or safe_retreat)

        info = {
            "conflict":          conflict_now,
            "conflict_imminent": conflict_imminent,
            "distance_to_farm":  distance_to_farmland,
            "distance_to_water": distance_to_water,
            "step":              self.current_step,
            "season":            "dry" if self.season == 1 else "wet",
            "conflict_history":  self.conflict_history,
            "is_night":          is_night,
        }

        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.draw(self.state, self.current_step, action)

        return self.state, reward, terminated, truncated, info

    # Reward function 
    def _calculate_reward(self, action, conflict_now, conflict_imminent,
                          distance_to_farmland):
        reward = 0.0

        if conflict_now:
           
            if action in (self.HIGH_ALERT, self.DEPLOY_RANGER, self.SEND_SMS):
                reward = +10.0
            elif action == self.ACTIVATE_DETERRENT:
                reward = +5.0   
            elif action == self.LOW_ALERT:
                reward = -3.0 
            else:           
                reward = -8.0  
        elif conflict_imminent:
          
            if action in (self.HIGH_ALERT, self.DEPLOY_RANGER, self.SEND_SMS):
                reward = +7.0
            elif action == self.ACTIVATE_DETERRENT:
                reward = +6.0
            elif action == self.LOW_ALERT:
                reward = +2.0
            else:               
                reward = -6.0

        elif distance_to_farmland < self.APPROACH_DISTANCE:
            if action == self.LOW_ALERT:
                reward = +3.0
            elif action == self.ACTIVATE_DETERRENT:
                reward = +4.0
            elif action == self.HIGH_ALERT:
                reward = +1.0   
            elif action in (self.DEPLOY_RANGER, self.SEND_SMS):
                reward = -1.0   
            else:              
                reward = -2.0

        else:
           
            if action == self.NO_ALERT:
                reward = +1.0
            elif action == self.LOW_ALERT:
                reward = -1.0
            elif action == self.ACTIVATE_DETERRENT:
                reward = -2.0
            elif action == self.HIGH_ALERT:
                reward = -4.0
            else:               
                reward = -5.0

        if (action == self.ACTIVATE_DETERRENT
                and not conflict_now
                and distance_to_farmland < self.APPROACH_DISTANCE):
            reward += 1.0

        return reward

    #  Render
    def render(self):
        
        if self.renderer is not None:
            self.renderer.draw(self.state, self.current_step)
        else:
        
            (step_length, speed, turning_angle,
             dist, ndvi, time_of_day, season,
             dist_water, conflict_hist) = self.state
            season_str = "DRY" if season == 1 else "WET"
            print(
                f"Step {self.current_step:03d} | "
                f"Dist: {dist:7.1f}m | "
                f"Water: {dist_water:7.1f}m | "
                f"Speed: {speed:.2f}m/s | "
                f"Season: {season_str} | "
                f"Time: {int(time_of_day):02d}:00 | "
                f"Conflicts: {int(conflict_hist)}"
            )

    # Closing
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None