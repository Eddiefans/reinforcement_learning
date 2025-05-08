import numpy as np
import gym
from gym import spaces

class TTLTradingEnv(gym.Env):
    """
    Gym environment for a TTL (Time-To-Live) trading agent:
    - Agent makes exactly one decision per TTL interval (5 days).
    - Decision is based on the previous week's (TTL) data.
    - Actions: 0 = DO NOTHING, 1 = BUY.
    - If DO NOTHING is chosen more than max_no_ops times total, a penalty is applied.
    - After each TTL interval, the environment advances by TTL days,
      resets position and capital for the next interval.
    - Tracks stats: wins, losses, average return, volatility.
    """

    def __init__(self, data, ttl = 5, window_size=20, initial_capital=1.0,
                 no_op_penalty=-0.01, no_op_mod=3):
        super().__init__()
        self.data = np.asarray(data)
        self.window_size = window_size
        self.ttl = ttl
        self.initial_capital = initial_capital
        self.no_op_penalty = no_op_penalty
        self.no_op_mod = no_op_mod

        self.total_no_ops = 0

        # Actions: 0 = HOLD (do nothing), 1 = BUY
        self.action_space = spaces.Discrete(2)

        # Observation: last `window_size` days of features
        n_features = self.data.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        # Initialize pointers and stats
        max_start = len(self.data) - self.window_size - self.ttl
        self.current_idx = np.random.randint(self.window_size, max_start)
        self.capital = self.initial_capital

        self.returns = []  # store each TTL return

        # Initial obs: window before decision point
        obs = self.data[self.current_idx - self.window_size : self.current_idx]
        return obs

    def step(self, action):
        # Fetch window and prices
        window = self.data[self.current_idx : self.current_idx + self.ttl]
        price_start = window[0, 0]
        price_end   = window[-1, 0]

        # Compute TTL return
        if action == 1:
            ret = np.log(price_end - price_start)
        else:
            ret = 0.0
            self.total_no_ops += 1
            if self.total_no_ops > self.max_no_ops:
                ret += self.no_op_penalty

        # Record return for stats
        self.returns.append(ret)

        # Reward is return × capital, then reset capital
        reward = ret * self.capital # We need to change the reward policy
        self.capital = self.initial_capital

        # Advance time
        self.current_idx += self.ttl
        done = (self.current_idx + self.ttl) >= len(self.data)

        if not done:
            obs = self.data[self.current_idx - self.window_size : self.current_idx]
        else:
            obs = np.zeros((self.window_size, self.data.shape[1]), dtype=np.float32)

        info = {
            "capital": self.capital,
            "total_no_ops": self.total_no_ops
        }
        return obs, reward, done, info

    def get_stats(self):
        """
        Returns dict with:
         - wins: number of positive-return TTLs
         - losses: number of negative-return TTLs
         - avg_return: mean of TTL returns
         - volatility: stddev of TTL returns
        """
        arr = np.array(self.returns, dtype=np.float32)
        wins = np.sum(arr > 0)
        losses = np.sum(arr < 0)
        avg_ret = float(np.mean(arr)) if arr.size else 0.0
        vol = float(np.std(arr)) if arr.size else 0.0
        total_no_ops = self.total_no_ops
        return {
            "wins": int(wins),
            "losses": int(losses),
            "avg_return": avg_ret,
            "volatility": vol,
            "total no ops": total_no_ops
        }
