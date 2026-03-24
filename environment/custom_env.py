import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WildlifeConflictEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    NO_ALERT   = 0
    LOW_ALERT  = 1
    HIGH_ALERT = 2

    CONFLICT_DISTANCE   = 200    
    IMMINENT_DISTANCE   = 1000  
    APPROACH_DISTANCE   = 3000   
    SAFE_RETREAT        = 9000   

    MAX_STEPS = 200

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        low  = np.array([0.0,  0.0,   0.0,  0.0, -1.0,  0.0, 0.0], dtype=np.float32)
        high = np.array([5000, 10.0, 180.0, 10000, 1.0, 23.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.state        = None
        self.current_step = 0
        self.season       = 0   
        self.conflict_occurred = False

        self.renderer = None

 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step      = 0
        self.conflict_occurred = False

        self.season = self.np_random.integers(0, 2)

        time_of_day = float(self.np_random.integers(0, 24))

        self.state = np.array([
            self.np_random.uniform(50,  300),    
            self.np_random.uniform(0.1, 1.0),    
            self.np_random.uniform(60,  180),    
            self.np_random.uniform(7000, 10000), 
            self.np_random.uniform(0.3, 1.0), 
            time_of_day,                        
            float(self.season),                 
        ], dtype=np.float32)

        info = {}
        return self.state, info

  
    def step(self, action):
        # Unpack current state
        (step_length, speed, turning_angle,
         distance_to_farmland, ndvi, time_of_day, season) = self.state

        if self.season == 1:  # dry
            step_length  += self.np_random.uniform(50,  250)
            speed        += self.np_random.uniform(0.1, 0.8)
        else:           
            step_length  += self.np_random.uniform(10,  80)
            speed        += self.np_random.uniform(0.0, 0.3)

        is_night = (time_of_day >= 18) or (time_of_day <= 6)
        if is_night:
            speed       += self.np_random.uniform(0.1, 0.4)
            step_length += self.np_random.uniform(20,  100)

        if distance_to_farmland < self.APPROACH_DISTANCE:
            turning_angle = max(5.0, turning_angle - self.np_random.uniform(5, 20))
        else:
            turning_angle = min(180.0, turning_angle + self.np_random.uniform(0, 10))

        if self.season == 1:
            ndvi = max(-1.0, ndvi - self.np_random.uniform(0.01, 0.05))
        else:
            ndvi = min(1.0,  ndvi + self.np_random.uniform(0.0,  0.02))

        move_toward_farm = speed * self.np_random.uniform(50, 150)
        distance_to_farmland = max(0.0, distance_to_farmland - move_toward_farm)

        time_of_day = (time_of_day + 0.5) % 24

        step_length          = np.clip(step_length,         0.0,  5000.0)
        speed                = np.clip(speed,               0.0,    10.0)
        turning_angle        = np.clip(turning_angle,       0.0,   180.0)
        distance_to_farmland = np.clip(distance_to_farmland, 0.0, 10000.0)
        ndvi                 = np.clip(ndvi,               -1.0,     1.0)

        # Update state
        self.state = np.array([
            step_length, speed, turning_angle,
            distance_to_farmland, ndvi, time_of_day, float(self.season)
        ], dtype=np.float32)

        conflict_now     = distance_to_farmland <= self.CONFLICT_DISTANCE
        conflict_imminent = distance_to_farmland <= self.IMMINENT_DISTANCE
        self.conflict_occurred = conflict_now

        reward = self._calculate_reward(action, conflict_now, conflict_imminent,
                                        distance_to_farmland)

        self.current_step += 1

        terminated = bool(conflict_now)

        safe_retreat = distance_to_farmland >= self.SAFE_RETREAT
        truncated    = bool(self.current_step >= self.MAX_STEPS or safe_retreat)

        info = {
            "conflict":           conflict_now,
            "conflict_imminent":  conflict_imminent,
            "distance_to_farm":   distance_to_farmland,
            "step":               self.current_step,
            "season":             "dry" if self.season == 1 else "wet",
        }

      
        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, info

    #  REWARD FUNCTION
  
    def _calculate_reward(self, action, conflict_now, conflict_imminent,
                          distance_to_farmland):
      
        reward = 0.0

        if conflict_now:
            if action == self.HIGH_ALERT:
                reward = +10.0   
            elif action == self.LOW_ALERT:
                reward = -3.0   
            else:
                reward = -8.0    

        elif conflict_imminent:
            if action == self.HIGH_ALERT:
                reward = +7.0  
            elif action == self.LOW_ALERT:
                reward = +2.0   
            else:
                reward = -6.0   

        elif distance_to_farmland < self.APPROACH_DISTANCE:
            if action == self.HIGH_ALERT:
                reward = -2.0  
            elif action == self.LOW_ALERT:
                reward = +3.0    
            else:
                reward = -1.0   

        else:
            
            if action == self.NO_ALERT:
                reward = +1.0    
            elif action == self.LOW_ALERT:
                reward = -1.0  
            else:
                reward = -5.0    

        return reward

    # RENDERING
   
    def render(self):
        """
        Delegates to rendering.py Pygame visualisation.
        If renderer is not attached, prints state to terminal.
        """
        if self.renderer is not None:
            self.renderer.draw(self.state, self.current_step)
        else:
            (step_length, speed, turning_angle,
             dist, ndvi, time_of_day, season) = self.state
            season_str = "DRY" if season == 1 else "WET"
            print(
                f"Step {self.current_step:03d} | "
                f"Dist: {dist:7.1f}m | "
                f"Speed: {speed:.2f}m/s | "
                f"Season: {season_str} | "
                f"Time: {int(time_of_day):02d}:00"
            )

    # Closing
   
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None