import gymnasium as gym

class FrozenLakeWrapper:
    def __init__(self, is_slippery=False, map_name="8x8"):
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name=map_name)
        self.action_space = self.env.action_space   # Discrete(4)
        self.observation_space = self.env.observation_space  # Discrete(n)
    
    def reset(self):
        result = self.env.reset(seed=1995)
        if isinstance(result, tuple):
            return result[0]
        return result
    
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        return result
    
    def render(self):
        self.env.render()
