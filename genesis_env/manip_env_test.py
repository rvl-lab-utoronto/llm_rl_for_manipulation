from franka_manip_env import FrankaManipEnv

env = FrankaManipEnv(render_video=False,show_viewer=True,verbose=True, use_gpu=True) # change however you want
goal_dict = {'red_cube_goal':[.25,.25,0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[.25,.5,0.02],
             'green_cube_goal':[0,.5,0.02]}
env.reset(goal_location=goal_dict)
# correct plan
llm_plan = 'move_x(-0.25)\nmove_y(0.25)\n\pick_block()\nmove_x(0.25)\nplace_block()'
#llm_plan = 'move_z(1.5)\nmove_z(-1.5)\nmove_z(1.5)'
reward = env.execute_llm_plan(llm_plan)
print('Correct Sequence Reward:', reward)
# incorrect plan
env.reset(goal_location=goal_dict)
llm_plan = 'move_x(-0.25)\nmove_y(0.5)'
reward = env.execute_llm_plan(llm_plan)
print('Incorrect Sequence Reward:', reward)
