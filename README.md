# Mission-Based Reinforcement Learning
### ULINZI Wildlife Conflict Early Warning

Ulinzi means *protection* in Swahili. This project applies reinforcement learning to a real problem I am researching for my capstone: predicting when a buffalo is about to leave a national park and raid nearby farmland, and issuing the right alert to farming communities before it happens.

The agent learns from a simulated environment built around actual buffalo movement patterns near Volcanoes National Park in Rwanda's Musanze District. Four RL algorithms are trained and compared on the same environment.

---

## Repository structure

```
Mission-Based-Reinforcement-Learning/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py     
│   └── rendering.py      
├── training/
│   ├── dqn_training.py     
│   └── pg_training.py     
├── models/
│   ├── dqn/                — saved DQN models, results CSV, plots
│   └── pg/                 — saved PPO and A2C models, results, plots
├── main.py                 — entry point to run the best trained agent
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/carine-ahishakiye/Mission-Based-Reinforcement-Learning.git
cd Mission-Based-Reinforcement-Learning

python -m venv venv
venv\Scripts\activate    for window
# source venv/bin/activate   for Linux or Mac

pip install -r requirements.txt
```

---

## Training

Train DQN (10 hyperparameter configurations):

```bash
python -m training.dqn_training
```

Train REINFORCE, PPO and A2C (10 configurations each):

```bash
python -m training.pg_training
```

Each script saves every model to `models/dqn/` or `models/pg/`, writes a results CSV with all 10 runs, and generates reward curves, entropy curves and a convergence plot automatically.

---

## Running the simulation

```bash
python main.py
```

This reads the summary JSON files from both `models/dqn/results/` and `models/pg/results/`, picks the algorithm with the highest mean evaluation reward, loads that model, opens a pygame window, and runs 3 episodes with full terminal output at each step.

To run a specific algorithm:

```bash
python main.py --algo ppo
python main.py --algo dqn
python main.py --algo a2c
python main.py --algo ppo --episodes 5
```

To also start a local JSON API at `http://localhost:5000/predict` that a web or mobile frontend can poll for live alert data:

```bash
python main.py --api
```

---

## Environment  WildlifeConflict-v0

The environment simulates a buffalo moving from a national park toward farmland across three zones. Each step represents 30 minutes of real time.

**Observation space  9 continuous features**

| Feature | Description |
|---|---|
| step_length | Distance moved between consecutive positions (metres) |
| speed | Movement speed (m/s) |
| turning_angle | Direction change in degrees  decreases as animal heads toward farmland |
| distance_to_farmland | Main risk indicator (metres) |
| ndvi | Vegetation index lower values mean food is scarce inside the park |
| time_of_day | Hour of day  buffalo move more at night |
| season | 0 = wet season, 1 = dry season |
| distance_to_water | Water availability drives movement decisions |
| conflict_history | Number of conflicts that occurred so far this episode |

**Action space  6 discrete actions**

| Action | When it earns a positive reward |
|---|---|
| NO ALERT | Buffalo is far from farmland, beyond 3000m |
| LOW ALERT | Buffalo entering the approach zone, 1000–3000m |
| HIGH ALERT | Buffalo is imminent, under 1000m |
| DEPLOY RANGER | Buffalo imminent or at the boundary |
| SEND SMS TO FARMERS | Buffalo at or within 200m of farmland |
| ACTIVATE DETERRENT | Buffalo approaching or imminent |

**Reward** shaped by distance zone and action choice. Correct actions earn +1 to +10. Wrong actions earn -1 to -8. The worst penalty is -8 for issuing no alert when the buffalo is already at the boundary.

**Terminal conditions**  an episode ends when the buffalo reaches the farmland boundary (distance ≤ 200m), retreats beyond 9000m, or after 200 steps.

---

## Results

| Algorithm | Best run | Mean eval reward |
|---|---|---|
| REINFORCE | Run 10 | 15.90 |
| A2C | Run 2 | 44.55 |
| PPO | Run 4 | **50.30** |
| DQN | see dqn_results.csv |  |

PPO performed best. It learned a clear escalation policy: stay quiet when the buffalo is far, activate the deterrent as it approaches, send SMS to farmers at the boundary.

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