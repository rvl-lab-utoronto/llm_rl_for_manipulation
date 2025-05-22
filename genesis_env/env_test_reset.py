from franka_manip_env import FrankaManipEnv

env = FrankaManipEnv(render_video=True,show_viewer=False,verbose=False, use_gpu=True) # change however you want

# note - manipulator starts at [0,0.25]
start_1_dict = {
            'red_cube_start':[0.25,0.25,0.02],
            'blue_cube_start': [-0.25,0.25,0.02],
            'yellow_cube_start': [0.25,0.5,0.02],
            'green_cube_start': [-0.25,0.5,0.02],

            'red_cube_goal':[0,0.375,0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[.25,.5,0.02],
             'green_cube_goal':[-0.25,0.5,0.02]}

start_2_dict = {
            'red_cube_start':[0.0,0.25,0.02],
            'blue_cube_start': [-0.25,0.25,0.02],
            'yellow_cube_start': [0.0,0.5,0.02],
            'green_cube_start': [-0.25,0.5,0.02],

            'red_cube_goal':[0,0.375,0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[.25,.5,0.02],
             'green_cube_goal':[-0.25,0.5,0.02]}

start_3_dict = {
            'red_cube_start':[0.3, 0.1, 0.02],
            'blue_cube_start': [0.3, 0.5, 0.02],
            'yellow_cube_start': [-0.15, 0.5, 0.02],
            'green_cube_start': [-0.15, 0.1, 0.02],

            'red_cube_goal':[0,0.375,0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[.25,.5,0.02],
             'green_cube_goal':[0,.5,0.02]}

start_dicts = [start_1_dict,start_2_dict,start_3_dict]

for start_dict in start_dicts:
    env.reset(task_dictionary=start_dict)
    llm_plan = '\nmove_x(0.25)'
    #llm_plan = 'move_z(1.5)\nmove_z(-1.5)\nmove_z(1.5)'
    reward = env.execute_llm_plan(llm_plan)
env.reset(task_dictionary=start_dict) # needed for video saving reasons 