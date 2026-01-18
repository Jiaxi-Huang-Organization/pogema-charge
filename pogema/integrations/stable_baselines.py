from typing import Optional
from gymnasium import Wrapper

class SingleAgentWrapper(Wrapper):
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(
            [action] + [self.env.action_space.sample() for _ in range(self.env.get_num_agents() - 1)])
        return observations[0], rewards[0], terminated[0], truncated[0], infos[0]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        result = self.env.reset(seed=seed, options=options)
        # gymnasium reset returns (observations, infos) tuple
        observations, infos = result
        return observations[0], infos[0]
def make_single_agent_gym(env):
    env = SingleAgentWrapper(env)
    return env