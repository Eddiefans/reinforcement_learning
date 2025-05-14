import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import your environment
from src.environment.trading_env import TTLTradingEnv

def train_model(
    data_path="data/processed/AMZN_features.csv",
    log_dir="logs/tensorboard",
    checkpoints_dir="checkpoints",
    best_model_dir="models/best_model",
    eval_log_dir="logs/evaluation",
    total_timesteps=50_000_000,
    ttl=10,
    window_size=1,
    no_op_penalty=-0.001,
    no_op_mod=3, 
    verbose=True
):
    """
    Train a PPO agent on the trading environment.
    
    Args:
        data_path: Path to the processed dataset
        log_dir: Directory for TensorBoard logs
        checkpoints_dir: Directory for model checkpoints
        best_model_dir: Directory to save the best model
        eval_log_dir: Directory for evaluation logs
        total_timesteps: Total number of timesteps to train for
        ttl: Time-to-live for trading decisions
        window_size: Observation window size
        no_op_penalty: Penalty for no-op actions
        no_op_mod: Apply penalty after this many no-ops
    
    Returns:
        model: Trained PPO model
    """
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Load data
    if verbose:
        print("Loading data from {}...".format(data_path))
    df = pd.read_csv(data_path)
    
    # Extract prices and features
    prices = df['Close'].astype(np.float32).values
    features = df.drop(columns=['Date', 'Close']).astype(np.float32).values
    
    # Create environments
    train_env = TTLTradingEnv(
        features=features,
        prices=prices,
        ttl=ttl,
        window_size=window_size,
        no_op_penalty=no_op_penalty,
        no_op_mod=no_op_mod
    )
    
    eval_env = TTLTradingEnv(
        features=features,
        prices=prices,
        ttl=ttl,
        window_size=window_size,
        no_op_penalty=no_op_penalty,
        no_op_mod=no_op_mod
    )
    
    # Configure logging
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    # Create model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
    )
    model.set_logger(new_logger)
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoints_dir,
        name_prefix="ppo_ttl"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    # Train model
    if verbose:
        print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="ppo_ttl_run"
    )
    
    # Save final model
    final_model_path = os.path.join("models", "ppo_ttl_final")
    model.save(final_model_path)
    print("Final model saved to {}".format(final_model_path))
    
    return model

if __name__ == "__main__":
    trained_model = train_model()