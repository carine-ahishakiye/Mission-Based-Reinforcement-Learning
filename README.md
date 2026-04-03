# ULINZI  Wildlife Conflict Early Warning System

*Ulinzi* means **protection** in Swahili.

This project trains a reinforcement learning agent to send the right wildlife conflict alert before a cape buffalo leaves a national park and enters farmland. The problem comes from pre-capstone research on human-wildlife conflict near Volcanoes National Park in Rwanda's Musanze District, where buffalo regularly destroy crops and hurt farmers in the Kinigi sector. GPS collar data shows that buffalo move differently hours before crossing a park boundary, but no system in Africa currently uses that signal to warn communities in time.

The agent observes nine features about the animal's movement and surroundings every 30 minutes, then picks one of six alert actions to send. Four algorithms were trained and compared on the same environment: DQN, PPO, A2C, and REINFORCE.

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
│   └── pg_training.py      # PPO, A2C, and REINFORCE with 10 runs each
├── models/
│   ├── dqn/                # saved DQN model, results JSON, plots
│   └── pg/                 # saved PPO, A2C, REINFORCE models, results, plots
├── main.py                 # entry point — loads best model and runs simulation
├── random_demo.py          # runs environment with random actions, no model needed
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
# source venv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

---

## How to run

**Random action demo (no model required):**
```bash
python random_demo.py
```
Opens the pygame window and runs episodes with random actions. Use this to verify the environment is set up correctly before any training.

**Train DQN (10 hyperparameter runs):**
```bash
python -m training.dqn_training
```
Saves the best model to `models/dqn/dqn_best.zip`, results to `models/dqn/dqn_results_summary.json`, and plots to `models/dqn/`.

**Train PPO, A2C, and REINFORCE (10 runs each):**
```bash
python -m training.pg_training
```
Loads PPO and A2C results from `models/pg/` if they exist, then trains REINFORCE. Saves best models and plots to `models/pg/`.

**Run the best trained agent:**
```bash
python main.py
```
Reads summary JSON files from `models/dqn/` and `models/pg/`, picks the algorithm with the highest mean evaluation reward (DQN at 33.275), loads that model, and runs 3 episodes with the pygame window and full per-step terminal output.

**Run a specific algorithm:**
```bash
python main.py --algo dqn
python main.py --algo ppo
python main.py --algo a2c
python main.py --algo reinforce
```

**Other options:**
```bash
python main.py --episodes 5 --seed 15   # 5 episodes, different seed
python main.py --headless               # no pygame window, terminal only
python main.py --random                 # random agent demo
```

---

## Environment: WildlifeConflict-v0

The environment simulates a buffalo moving from a national park toward farmland. Each step represents 30 minutes of simulated time and episodes run up to 48 steps (24 hours). In 60% of episodes a crossing is scheduled at a random step between step 4 and step 44. The other 40% are quiet episodes with no crossing.

**Observation space — 9 continuous features**

| Feature | Range | Description |
|---|---|---|
| boundary_proximity | 0 to 1 | Normalised distance to the park boundary |
| speed_norm | 0 to 1 | Movement speed |
| heading_change_norm | 0 to 1 | Lower values mean heading directly toward the boundary |
| displacement_x_norm | -1 to 1 | Lateral displacement |
| displacement_y_norm | -1 to 1 | Longitudinal displacement |
| time_of_day_norm | -1 to 1 | Sine-encoded hour; buffalo cross more often at night |
| vegetation_density | 0 to 1 | Low vegetation correlates with movement toward farmland |
| herd_cohesion | 0 to 1 | Tight herds move with more purpose |
| recent_crossing_history | 0 to 1 | Fraction of the last 6 steps with an active alert |

**Action space — 6 discrete actions**

| ID | Label | When it earns positive reward |
|---|---|---|
| 0 | NO ALERT | Risk is low, boundary proximity is low |
| 1 | COMMUNITY SMS | Moderate or imminent risk |
| 2 | RANGER DISPATCH | Crossing imminent |
| 3 | SCARE DEVICE ACTIVATE | Crossing imminent |
| 4 | ELEVATED WATCH | Moderate risk |
| 5 | EMERGENCY BROADCAST | Crossing imminent |

**Reward structure**

| Situation | Action | Reward |
|---|---|---|
| Crossing imminent (within 6 steps) | EMERGENCY BROADCAST | +5.0 |
| Crossing imminent | RANGER DISPATCH | +4.0 |
| Crossing imminent | SCARE DEVICE | +3.5 |
| Crossing imminent | COMMUNITY SMS | +2.0 |
| Crossing imminent | ELEVATED WATCH | +1.5 |
| Crossing imminent | NO ALERT | **-8.0** |
| High risk (score > 0.65) | COMMUNITY SMS or ELEVATED WATCH | +2.5 |
| High risk | NO ALERT | -3.0 |
| High risk | EMERGENCY BROADCAST | -1.5 |
| Low risk (score < 0.3) | NO ALERT | +1.0 |
| Low risk | EMERGENCY BROADCAST | -4.0 |
| Low risk | RANGER DISPATCH or SCARE DEVICE | -2.0 |
| Low risk | COMMUNITY SMS or ELEVATED WATCH | -0.5 |

Risk score = 0.5 × boundary_proximity + 0.3 × speed_norm + 0.2 × herd_cohesion

**Terminal conditions:** episode ends when the buffalo crosses the boundary or after 48 steps.

---

## Results

| Algorithm | Best run | Mean eval reward | Std |
|---|---|---|---|
| DQN | Run 4 | **33.275** | 26.623 |
| PPO | Run 7 | 21.125 | 19.174 |
| A2C | Run 3 | 17.250 | 19.683 |
| REINFORCE | Run 4 | -0.600 | 5.798 |

**DQN** achieved the highest mean evaluation reward. Run 4 used `lr=1e-3`, `gamma=0.90`, `batch_size=128`, and `target_update_interval=250`. The lower discount factor (0.90 vs the typical 0.99) helped by weighting near-term rewards more heavily, which suits a 48-step episode where the critical crossing window only opens in the final 6 steps. Experience replay was the core advantage over on-policy methods: DQN revisits rare crossing-window transitions many times per training step, while PPO and A2C see each one once and discard it.

**PPO** was the most robust policy gradient method. Every one of its 10 runs finished positive. Run 7 succeeded by using a higher entropy coefficient (0.05) that kept the policy exploring long enough to discover the full escalation sequence.

**A2C** was the most consistent algorithm overall: 9 out of 10 runs finished positive. Run 3 used `gamma=0.95` and a reduced value function coefficient (0.25), letting the actor gradient dominate early training.

**REINFORCE** did not converge in any run. Short episodes, a narrow reward window, and high-variance gradient estimates are a difficult combination. Run 4 came closest at -0.60 by using a high entropy coefficient (0.05) and a value baseline.

---

## Best hyperparameter configurations

**DQN Run 4 (best)**
```
learning_rate:          1e-3
gamma:                  0.90
batch_size:             128
buffer_size:            20000
exploration_fraction:   0.20
target_update_interval: 250
```

**PPO Run 7 (best)**
```
learning_rate: 3e-4
gamma:         0.99
n_steps:       128
batch_size:    64
n_epochs:      10
ent_coef:      0.05
clip_range:    0.2
gae_lambda:    0.95
```

**A2C Run 3 (best)**
```
learning_rate:  5e-4
gamma:          0.95
n_steps:        10
ent_coef:       0.02
vf_coef:        0.25
max_grad_norm:  0.5
gae_lambda:     0.9
```

**REINFORCE Run 4 (best)**
```
learning_rate: 2e-3
gamma:         0.99
hidden_size:   64
entropy_coef:  0.05
baseline:      True
```

---

## Dependencies

```
gymnasium==0.29.1
stable-baselines3==2.8.0
torch>=2.0.0
numpy>=1.24.0
pygame>=2.5.0
matplotlib>=3.7.0
shimmy>=0.2.0
```