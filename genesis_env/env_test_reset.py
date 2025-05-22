from franka_manip_env import FrankaManipEnv

env = FrankaManipEnv(render_video=False,show_viewer=False,verbose=False, use_gpu=True) # change however you want

# note - manipulator starts at [0,0.25]
red_dict = {
            'red_cube_start':[0.25,0.25,0.02],
            'blue_cube_start': [-0.25,0.25,0.02],
            'yellow_cube_start': [0.25,0.5,0.02],
            'green_cube_start': [-0.25,0.5,0.02],

            'red_cube_goal':[0,0.375,0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[.25,.5,0.02],
             'green_cube_goal':[0,.5,0.02]}

green_dict = {
            'red_cube_start':[0.25,0.25,0.02],
            'blue_cube_start': [-0.25,0.25,0.02],
            'yellow_cube_start': [0.25,0.5,0.02],
            'green_cube_start': [-0.25,0.5,0.02],

            'red_cube_goal':[0.25,0.25,0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[.25,.5,0.02],
             'green_cube_goal':[0,0.375,0.02]}

blue_dict = {
            'red_cube_start':[0.25,0.25,0.02],
            'blue_cube_start': [-0.25,0.25,0.02],
            'yellow_cube_start': [0.25,0.5,0.02],
            'green_cube_start': [-0.25,0.5,0.02],

            'red_cube_goal':[0.25,0.25,0.02],
             'blue_cube_goal':[0,0.375,0.02],
             'yellow_cube_goal':[.25,.5,0.02],
             'green_cube_goal':[-0.25,0.5,0.02]}

yellow_dict = {
            'red_cube_start':[0.25,0.25,0.02],
            'blue_cube_start': [-0.25,0.25,0.02],
            'yellow_cube_start': [0.25,0.5,0.02],
            'green_cube_start': [-0.25,0.5,0.02],

            'red_cube_goal':[0.25,0.25,0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[0,0.375,0.02],
             'green_cube_goal':[-0.25,0.5,0.02]}
env.reset(task_dictionary=red_dict)

# correct plan(s)
print('Red Block Manipulation Check')
correct = 0
for i in range(8):
    llm_plan = '\nmove_x(0.25)\npick_block()\nmove_x(-0.25)\nmove_y(0.125)\nplace_block()'
    #llm_plan = 'move_z(1.5)\nmove_z(-1.5)\nmove_z(1.5)'
    reward = env.execute_llm_plan(llm_plan)
    correct += reward
    #print('Correct Sequence Reward:', reward)
    env.reset(task_dictionary=red_dict)
print('Red Correct:',correct,'/ 8')

# correct plan(s)
print('Green Block Manipulation Check')
env.reset(task_dictionary=green_dict)
correct = 0
for i in range(8):
    llm_plan = '\nmove_x(-0.25)\nmove_y(0.25)\npick_block()\nmove_x(0.25)\nmove_y(-0.125)\nplace_block()'
    #llm_plan = 'move_z(1.5)\nmove_z(-1.5)\nmove_z(1.5)'
    reward = env.execute_llm_plan(llm_plan)
    correct += reward
    #print('Correct Sequence Reward:', reward)
    env.reset(task_dictionary=green_dict)
print('Green Correct:',correct,'/ 8')

# correct plan(s)
print('Yellow Block Manipulation Check')
env.reset(task_dictionary=yellow_dict)
correct = 0
for i in range(8):
    llm_plan = '\nmove_x(0.25)\nmove_y(0.25)\npick_block()\nmove_x(-0.25)\nmove_y(-0.125)\nplace_block()'
    #llm_plan = 'move_z(1.5)\nmove_z(-1.5)\nmove_z(1.5)'
    reward = env.execute_llm_plan(llm_plan)
    correct += reward
    #print('Correct Sequence Reward:', reward)
    env.reset(task_dictionary=yellow_dict)
print('Yellow Correct:',correct,'/ 8')

# correct plan(s)
print('Blue Block Manipulation Check')
env.reset(task_dictionary=blue_dict)
correct = 0
for i in range(8):
    llm_plan = '\nmove_x(-0.25)\npick_block()\nmove_x(0.25)\nmove_y(0.125)\nplace_block()'
    #llm_plan = 'move_z(1.5)\nmove_z(-1.5)\nmove_z(1.5)'
    reward = env.execute_llm_plan(llm_plan)
    correct += reward
    #print('Correct Sequence Reward:', reward)
    env.reset(task_dictionary=blue_dict)
print('Blue Correct:',correct,'/ 8')

# incorrect plan
print('Incorrect Plan Test:')
env.reset(task_dictionary=red_dict)
llm_plan = 'move_x(-0.25)\nmove_y(0.5)'
reward = env.execute_llm_plan(llm_plan)
print('Incorrect Sequence Reward:', reward)
