from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
import numpy as np
from trading_env import TTLTradingEnv  # assuming the env code is in ttl_trading_env.py

# Load your dataset: numpy array of shape (T, features)
data = np.load("stock_data.npy")

# Create training and evaluation environments
train_env = TTLTradingEnv(data)
eval_env = TTLTradingEnv(data)

# Configure tensorboard logging
log_dir = "./ppo_ttl_tensorboard/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# Define the PPO model
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log=log_dir,
)

model.set_logger(new_logger)

# Checkpoint callback: save every 10000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints/",
    name_prefix="ppo_ttl"
)

# Evaluation callback: every 5000 steps, evaluate for 1000 timesteps
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=5_000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

# Train the agent
model.learn(
    total_timesteps=50_000_000,
    callback=[checkpoint_callback, eval_callback],
    tb_log_name="ppo_ttl_run"
)

# Save the final model
model.save("ppo_ttl_last_model")
