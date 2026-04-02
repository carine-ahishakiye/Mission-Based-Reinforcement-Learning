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
│   ├── custom_env.py       — WildlifeConflict-v0 custom Gymnasium environment
│   └── rendering.py        — Pygame visualisation with three zones, buffalo, action banner
├── training/
│   ├── dqn_training.py     — DQN training with 10 hyperparameter runs
│   └── pg_training.py      — REINFORCE, PPO, and A2C with 10 runs each
├── models/
│   ├── dqn/                — saved DQN models, results CSV, plots
│   └── pg/                 — saved PPO and A2C models, results, plots
├── main.py                 
├── random_demo.py          — show the environment with random actions, no model
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/carine-ahishakiye/Mission-Based-Reinforcement-Learning.git
cd Mission-Based-Reinforcement-Learning

python -m venv venv
venv\Scripts\activate       
# source venv/bin/activate 

pip install -r requirements.txt
```

---

## How to run

**Random action demo no model required:**
```bash
python random_demo.py
```
Opens the pygame window and runs 3 episodes with random actions. Shows the environment components working without any training.

**Train DQN (10 hyperparameter runs):**
```bash
python -m training.dqn_training
```

**Train REINFORCE, PPO, and A2C (10 runs each):**
```bash
python -m training.pg_training
```

Both scripts save every model, export results to CSV, and generate reward curves, entropy curves, and convergence plots automatically in `models/dqn/plots/` and `models/pg/plots/`.

**Run the best trained agent:**
```bash
python main.py
```

Reads the summary JSON files from both results folders, picks the algorithm with the highest mean evaluation reward, loads that model, and runs 3 episodes with the pygame window and full terminal output per step.

Run a specific algorithm:
```bash
python main.py --algo ppo
python main.py --algo dqn --episodes 5
python main.py --algo a2c
```

Start the JSON API alongside the simulation so any web or mobile frontend can poll live alert data:
```bash
python main.py --api
```
Each step is serialized to JSON at `http://localhost:5000/predict`.

---

## Environment: WildlifeConflict-v0

The environment simulates a buffalo moving from a national park toward farmland across three zones. Each step represents 30 minutes of simulated time.

**Observation space  9 continuous features**

| Feature | Range | Description |
|---|---|---|
| step_length | 0 to 5000m | Distance moved between GPS readings |
| speed | 0 to 10 m/s | Movement speed |
| turning_angle | 0 to 180 degrees | Lower values mean heading straight toward farmland |
| distance_to_farmland | 0 to 10000m | Main conflict risk signal |
| ndvi | -1 to 1 | Vegetation quality lower means food is scarce |
| time_of_day | 0 to 23 | Buffalo move more at night |
| season | 0 or 1 | Dry season means faster and further movement |
| distance_to_water | 0 to 10000m | Water scarcity drives movement |
| conflict_history | 0 to 10 | Cumulative conflicts this episode |

**Action space 6 discrete actions**

| Action | When it earns positive reward |
|---|---|
| 0 - NO ALERT | Buffalo is far, beyond 3000m |
| 1 - LOW ALERT | Buffalo entering approach zone, 1000 to 3000m |
| 2 - HIGH ALERT | Buffalo imminent, under 1000m |
| 3 - DEPLOY RANGER | Conflict imminent or at boundary |
| 4 - SEND SMS TO FARMERS | Buffalo at or within 200m of farmland |
| 5 - ACTIVATE DETERRENT | Approach or imminent zone |

**Reward structure**

| Zone | Best action | Reward |
|---|---|---|
| Far (beyond 3000m) | NO ALERT | +1 |
| Approach (1000 to 3000m) | ACTIVATE DETERRENT | +4 |
| Imminent (under 1000m) | HIGH ALERT / DEPLOY / SMS | +7 |
| Boundary (under 200m) | HIGH ALERT / DEPLOY / SMS | +10 |
| Boundary with NO ALERT | NO ALERT | -8 |

**Terminal conditions**

An episode ends when the buffalo reaches farmland (conflict, distance 200m or less), retreats beyond 9000m, or after 200 steps maximum.

---

## Results

| Algorithm  | Best run | Mean eval reward |
|------------|----------|------------------|
| REINFORCE  | Run 7    | -124.95          |
| DQN        | Run 5    | 348.55           |
| A2C        | Run 2    | 274.28           |
| PPO        | Run 8    | **311.30**       |

DQN achieved the highest mean evaluation reward in its best run, showing strong performance with the right hyperparameters.  
PPO was more stable across runs and learned a clear escalation policy: stay quiet when the buffalo is far, activate the deterrent in the approach zone, send SMS to farmers in the imminent zone, and deploy rangers at the boundary.  
A2C was consistent and competitive, while REINFORCE struggled with high variance and most runs finished negative.


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
  "action": 5,
  "action_name": "ACTIVATE DETERRENT",
  "reward": 5.0,
  "distance_to_farm": 1463.9,
  "speed_ms": 1.84,
  "season": "dry",
  "is_night": true,
  "conflict_now": false,
  "conflict_imminent": false,
  "conflict_history": 0

}
```

Any web or mobile frontend can poll this endpoint to display real-time conflict alerts.

---

## Dependencies

```
gymnasium==0.29.1
stable-baselines3==2.3.2
torch==2.2.2
numpy==1.26.4
pandas==2.2.2
pygame==2.5.2
matplotlib==3.8.4
tqdm==4.66.4
```

---
