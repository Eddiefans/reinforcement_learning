import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from trading_env import TTLTradingEnv

# 1) Load CSV
df = pd.read_csv("/Users/alexsotomayor/code/trading/reinforcement_learning/amzn_dataset.csv")

# 2) Split out prices and features, drop Date & Close from features
prices = df['Close'].astype(np.float32).values
features = df.drop(columns=['Date', 'Close']).astype(np.float32).values

# 3) Instantiate training and evaluation environments
train_env = TTLTradingEnv(
    features,
    prices,
    ttl=5,
    window_size=1,
    no_op_penalty=-0.005,
    no_op_mod=3
)
eval_env = TTLTradingEnv(
    features=features,
    prices=prices,
    window_size=1,
    ttl=5,
    no_op_penalty=-0.005,
    no_op_mod=3
)

# 4) Configure TensorBoard logging
log_dir = "./ppo_ttl_tensorboard/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# 5) Define the PPO model
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log=log_dir,
)
model.set_logger(new_logger)

# 6) Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints/",
    name_prefix="ppo_ttl"
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=5_000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

# 7) Train
model.learn(
    total_timesteps=50_000_000,
    callback=[checkpoint_callback, eval_callback],
    tb_log_name="ppo_ttl_run"
)

# 8) Save the final model
model.save("ppo_ttl_last_model")
