import csv, os

save_dir = r"C:\Users\PC\Desktop\LTP\Mission-Based-Reinforcement-Learning\models\dqn"
os.makedirs(save_dir, exist_ok=True)

rows = [
    {"run": 1,  "learning_rate": "1.0e-03", "gamma": 0.99, "batch_size": 64,  "buffer_size": 10000,  "exploration_fraction": 0.20, "mean_reward": 13.400, "std_reward": 18.421},
    {"run": 2,  "learning_rate": "5.0e-04", "gamma": 0.99, "batch_size": 64,  "buffer_size": 50000,  "exploration_fraction": 0.30, "mean_reward": 12.325, "std_reward": 18.377},
    {"run": 3,  "learning_rate": "1.0e-04", "gamma": 0.95, "batch_size": 32,  "buffer_size": 10000,  "exploration_fraction": 0.40, "mean_reward": 18.550, "std_reward": 20.032},
    {"run": 4,  "learning_rate": "1.0e-03", "gamma": 0.90, "batch_size": 128, "buffer_size": 20000,  "exploration_fraction": 0.20, "mean_reward": 33.275, "std_reward": 26.623},
    {"run": 5,  "learning_rate": "2.0e-04", "gamma": 0.99, "batch_size": 64,  "buffer_size": 50000,  "exploration_fraction": 0.50, "mean_reward": 13.800, "std_reward": 18.915},
    {"run": 6,  "learning_rate": "5.0e-03", "gamma": 0.95, "batch_size": 32,  "buffer_size": 10000,  "exploration_fraction": 0.30, "mean_reward": 15.200, "std_reward": 25.105},
    {"run": 7,  "learning_rate": "1.0e-04", "gamma": 0.99, "batch_size": 128, "buffer_size": 100000, "exploration_fraction": 0.20, "mean_reward": 22.025, "std_reward": 20.148},
    {"run": 8,  "learning_rate": "3.0e-04", "gamma": 0.97, "batch_size": 64,  "buffer_size": 20000,  "exploration_fraction": 0.35, "mean_reward": 14.000, "std_reward": 18.753},
    {"run": 9,  "learning_rate": "1.0e-03", "gamma": 0.99, "batch_size": 256, "buffer_size": 50000,  "exploration_fraction": 0.25, "mean_reward": 15.250, "std_reward": 21.986},
    {"run": 10, "learning_rate": "6.0e-04", "gamma": 0.98, "batch_size": 64,  "buffer_size": 30000,  "exploration_fraction": 0.30, "mean_reward": 11.950, "std_reward": 20.478},
]

save_path = os.path.join(save_dir, "dqn_summary_table.csv")
with open(save_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {save_path}")