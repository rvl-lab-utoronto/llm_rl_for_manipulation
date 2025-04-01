import numpy as np
import ogbench
import gymnasium as gym
from environment_utils import PrimitiveActionWrapper, SceneDict, save_numpy_array_as_video



env_name = 'visual-cube-single-play-v0'
env = PrimitiveActionWrapper(ogbench.make_env_and_datasets(env_name, env_only=True))


action_traj = [
    {
      "primitive": 0,
      "magnitude": 1  
    },
    {
        "primitive": 0,
      "magnitude": 1
    },
    {
        "primitive": 0,
      "magnitude": 1
    },
    {
        "primitive": 1,
      "magnitude": 1
    },
    {
        "primitive": 1,
      "magnitude": 1
    },
    {
        "primitive": 1,
      "magnitude": 1
    },
    {
        "primitive": 1,
      "magnitude": 1
    },
    {
        "primitive": 1,
      "magnitude": 1
    },
    {
        "primitive": 2,
      "magnitude": 1
    },
    {
        "primitive": 2,
      "magnitude": 1
    },
    {
        "primitive": 2,
      "magnitude": 1
    },
]

env.reset(seed=1)
obs = []
for d in action_traj:
    s, _, _, _, _ = env.step(d)
    
    obs.append(s)
print(obs[-1].shape)

save_numpy_array_as_video(np.array(obs), output_path='testing_results/brute_force_path.mp4', fps=2)

