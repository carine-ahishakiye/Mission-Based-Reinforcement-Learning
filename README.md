# ULINZI Wildlife Conflict Early Warning System

*Ulinzi* means **protection** in Swahili.

This project trains a reinforcement learning agent to send the right wildlife conflict alert before a cape buffalo leaves a national park and enters farmland. The problem comes from my pre-capstone research on human-wildlife conflict near Volcanoes National Park in Rwanda's Musanze District, where buffalo regularly destroy crops and hurt farmers in the Kinigi sector. GPS collar data shows that buffalo move differently hours before crossing a park boundary, but no system in Africa currently uses that signal to warn communities in time.

The agent observes nine features about the animal's movement and surroundings every 30 minutes, then picks one of six alert actions to send. Four algorithms were trained and compared on the same environment: DQN, REINFORCE, PPO, and A2C.

---

## Project structure

```
Mission-Based-Reinforcement-Learning/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py       # WildlifeConflict-v0 custom Gymnasium environment
│   └── rendering.py        # Pygame visualisation 
├── training/
│   ├── dqn_training.py     # DQN with 10 hyperparameter runs
│   └── pg_training.py      # REINFORCE, PPO, and A2C with 10 runs each
├── models/
│   ├── dqn/                # saved DQN models, results JSON, plots
│   └── pg/                 # saved PPO, A2C, and REINFORCE models, results, plots
├── main.py
├── random_demo.py          # run the environment with random actions, no model needed
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/carine-ahishakiye/Mission-Based-Reinforcement-Learning.git
cd Mission-Based-Reinforcement-Learning

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

---

## How to run

**Random action demo (no model required):**
```bash
python random_demo.py
```
Opens the pygame window and runs 3 episodes with random actions. Good for checking the environment is set up correctly before any training.

**Train DQN (10 hyperparameter runs):**
```bash
python -m training.dqn_training
```

**Train REINFORCE, PPO, and A2C (10 runs each):**
```bash
python -m training.pg_training
```

Note: `pg_training.py` loads PPO and A2C results from `models/pg/` before running REINFORCE, so run the PPO and A2C training first or generate the JSON files manually.

Both scripts save every model, export results to JSON, and generate reward curves, entropy curves, and convergence plots in `models/dqn/` and `models/pg/`.

**Run the best trained agent:**
```bash
python main.py
```

Reads the summary JSON files from both results folders, picks the algorithm with the highest mean evaluation reward, loads that model, and runs 3 episodes with the pygame window open and full per-step terminal output.

Run a specific algorithm:
```bash
python main.py --algo ppo
python main.py --algo dqn --episodes 5
python main.py --algo a2c
python main.py --algo reinforce
```

Run headless (no pygame window):
```bash
python main.py --headless
```

Start the JSON API alongside the simulation so any web or mobile frontend can poll live alert data:
```bash
python main.py --api
```
Each step is serialized to JSON at `http://localhost:5000/predict`.

---

## Environment: WildlifeConflict-v0

The environment simulates a buffalo moving from a national park toward farmland. Each step represents 30 minutes of simulated time, and episodes run up to 48 steps.

**Observation space: 9 continuous features**

| Feature | Range | Description |
|---|---|---|
| boundary_proximity | 0 to 1 | Normalized distance to the park boundary |
| speed_norm | 0 to 1 | Movement speed |
| heading_change_norm | 0 to 1 | Lower values mean heading more directly toward the boundary |
| displacement_x_norm | -1 to 1 | Lateral displacement |
| displacement_y_norm | -1 to 1 | Longitudinal displacement |
| time_of_day_norm | -1 to 1 | Sine-encoded hour; buffalo move more at night |
| vegetation_density | 0 to 1 | Low vegetation pushes buffalo toward farmland |
| herd_cohesion | 0 to 1 | How tightly grouped the herd is |
| recent_crossing_history | 0 to 1 | Proportion of recent steps with active alerts |

**Action space: 6 discrete actions**

| Action | Label | When it earns positive reward |
|---|---|---|
| 0 | NO ALERT | Risk is low, boundary proximity is low |
| 1 | COMMUNITY SMS | Moderate or imminent risk |
| 2 | RANGER DISPATCH | Crossing imminent |
| 3 | SCARE DEVICE ACTIVATE | Crossing imminent |
| 4 | ELEVATED WATCH | Moderate risk |
| 5 | EMERGENCY BROADCAST | Crossing imminent |

**Reward structure**

| Situation | Best action | Reward |
|---|---|---|
| Crossing imminent (within 6 steps) | EMERGENCY BROADCAST | +5.0 |
| Crossing imminent | RANGER DISPATCH | +4.0 |
| Crossing imminent | SCARE DEVICE | +3.5 |
| Crossing imminent | COMMUNITY SMS | +2.0 |
| Crossing imminent | NO ALERT | -8.0 |
| High risk (score > 0.65) | COMMUNITY SMS or ELEVATED WATCH | +2.5 |
| High risk | NO ALERT | -3.0 |
| Low risk (score < 0.3) | NO ALERT | +1.0 |
| Low risk | EMERGENCY BROADCAST | -4.0 |
| Low risk | RANGER DISPATCH or SCARE DEVICE | -2.0 |

**Terminal conditions**

An episode ends when the buffalo crosses the boundary, or after 48 steps.

---

## Results

| Algorithm | Best run | Mean eval reward |
|---|---|---|
| REINFORCE | Run 5 | -7.750 |
| A2C | Run 3 | 17.250 |
| PPO | Run 7 | 21.125 |
| DQN | Run 4 | **33.275** |

DQN achieved the highest mean evaluation reward, with Run 4 finding a strong configuration: lr=0.001, gamma=0.90, batch size 128, and a fast target update interval of 250. PPO was the most consistent policy gradient method and learned a clear escalation policy: stay quiet when the buffalo is far, raise elevated watch as risk grows, and escalate to rangers or an emergency broadcast when a crossing is imminent. A2C was competitive in its best runs, while REINFORCE had high variance and all 10 runs finished negative.

The lower discount factor (gamma=0.90) in DQN's best run likely helped by weighting immediate reward signals more heavily, which suits a short 48-step episode where the critical window for acting is narrow. REINFORCE struggled most, since high-variance gradient estimates compound quickly with so few steps per episode.

---

## JSON API

Running `python main.py --api` starts a lightweight HTTP server. Each simulation step is available at:

```
GET http://localhost:5000/predict
```

Example response:
```json
{
  "step": 12,
  "action": 3,
  "action_name": "SCARE DEVICE ACTIVATE",
  "reward": 3.5,
  "boundary_proximity": 0.83,
  "risk_score": 0.71,
  "crossing_imminent": true,
  "false_alarms": 0,
  "successful_alerts": 2,
  "total_reward": 14.5
}
```

---

## Dependencies

```
gymnasium==0.29.1
stable-baselines3==2.3.2
torch==2.2.2
numpy==1.26.4
pygame==2.5.2
matplotlib==3.8.4
```