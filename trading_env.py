import numpy as np
import gym
from gym import spaces

class TTLTradingEnv(gym.Env):
    """
    Gym environment for a TTL (Time-To-Live) trading agent:
    - Agent makes exactly one decision per TTL interval (5 days).
    - Decision is based on the previous week's (TTL) data excluding price in the observation.
    - Actions: 0 = DO NOTHING, 1 = BUY.
    - If DO NOTHING is chosen more than max_no_ops times total, a penalty is applied.
    - After each TTL interval, the environment advances by TTL days,
      resets position and capital for the next interval.
    - Tracks stats: wins, losses, average return, volatility, total no-ops.
    - Hides the 'Close' price column (second column) in observations; uses it only for log-return rewards.
    """

    def __init__(self, features: np.ndarray, prices: np.ndarray, ttl=5, window_size=1,
                 no_op_penalty=-0.005, no_op_mod=3):
        super().__init__()
        assert len(features) == len(prices), "mismatched lengths"
        self.features = features
        self.prices = prices
        self.window_size = window_size
        self.ttl = ttl
        self.no_op_penalty = no_op_penalty
        self.no_op_mod = no_op_mod

        self.total_no_ops = 0
        self.returns = []

        # Actions: 0 = HOLD, 1 = BUY
        self.action_space = spaces.Discrete(2)

        n_feats = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, n_feats),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        max_start = len(self.features) - self.window_size - self.ttl
        self.current_idx = np.random.randint(self.window_size, max_start)
        self.returns.clear()
        return self._get_obs()

    def step(self, action):
        start = self.current_idx
        end = start + self.ttl - 1
        price_start = self.prices[start]
        price_end = self.prices[end]
        if action == 1:
            ret = np.log(price_end / price_start)
        else:
            ret = 0.0
            possible_ret = np.log(price_end / price_start)
            self.total_no_ops += 1
            if self.total_no_ops > self.no_op_mod and possible_ret > 0:
                ret += self.no_op_penalty
            elif possible_ret < 0:
                ret += -self.no_op_penalty

        self.returns.append(ret)
        reward = ret

        # Advance
        self.current_idx += self.ttl
        done = (self.current_idx + self.ttl) > len(self.features)
        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        info = {"total_no_ops": self.total_no_ops}
        return obs, reward, done, info

    def _get_obs(self):
        i = self.current_idx
        return self.features[i - self.window_size : i]

    def get_stats(self):
        arr = np.array(self.returns, dtype=np.float32)
        wins = int((arr > 0).sum())
        losses = int((arr < 0).sum())
        avg_ret = float(arr.mean()) if arr.size else 0.0
        vol = float(arr.std()) if arr.size else 0.0
        return {
            "wins": wins,
            "losses": losses,
            "avg_return": avg_ret,
            "volatility": vol,
            "total_no_ops": self.total_no_ops
        }
