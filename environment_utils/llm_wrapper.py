import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper, Wrapper
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv


class PrimitiveActionWrapper(ActionWrapper):
    def __init__(self, env):
        
        super().__init__(env)
        
        self.original_action_dim = env.action_space.shape[-1]
        self.original_action_scales = env.action_space.high, env.action_space.low
        self.action_space = gym.spaces.Dict(
            {
                "primitive": gym.spaces.Discrete(self.original_action_dim),
                "magnitude": gym.spaces.Box(-1,1, shape=(1,))
            }
        )
    
    def action(self, act):
        index = act['primitive']
        magnitude = act['magnitude']
        
        scaled_magnitude = (magnitude + 1)*(self.original_action_scales[0][index] - self.original_action_scales[1][index]) - self.original_action_scales[1][index]
        
        vector = np.zeros((self.original_action_dim,))
        vector[index] = magnitude
        
        return vector  
    
class SceneDict(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        env.reset()
        
        obs_info = env.unwrapped.compute_ob_info() 
        
        self.observation_space = gym.spaces.Dict(
            
            {key: gym.spaces.Box(-np.inf, np.inf, x.shape) for key, x in obs_info.items() if key != ['control','time']}
            
        )
    def observation(self, observation):
        
        
        
        obs_dict = self.env.unwrapped.compute_ob_info()
        
        obs_dict.pop('control')
        obs_dict.pop('time')
        return obs_dict
