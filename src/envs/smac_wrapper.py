from smac.env import StarCraft2Env

from .multiagentenv import MultiAgentEnv


class SMACWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, **kwargs):
        self.env = StarCraft2Env(map_name=map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit

    def step(self, actions, MultiObject=True):
        """Returns obss, reward, terminated, truncated, info"""
        # 调用环境的 step 方法
        result = self.env.step(actions, MultiObject=MultiObject)
        
        # 处理返回值数量不一致的情况
        if len(result) == 3:
            # 异常情况（如连接错误）：只返回 reward, terminated, info
            rews, terminated, info = result
            fitness = [0, 0, 0]  # 默认空适应度
        elif len(result) == 4:
            # 正常情况：返回 reward, terminated, info, fitness
            rews, terminated, info, fitness = result
        else:
            raise ValueError(f"Unexpected return values from env.step: got {len(result)} values")
        
        obss = self.get_obs()
        truncated = False
        return obss, rews, terminated, truncated, info, fitness

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        obss, _ = self.env.reset()
        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()
