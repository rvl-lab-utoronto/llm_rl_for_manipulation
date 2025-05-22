from franka_manip_env import FrankaManipEnv

env = FrankaManipEnv(render_video=False,show_viewer=False,verbose=False, use_gpu=True) # change however you want

# note - manipulator starts at [0,0.25]
edge_start_dict = {
            'red_cube_start':[-0.15, 0.1, 0.02],
            'blue_cube_start': [-0.25,0.25,0.02],
            'yellow_cube_start': [0.25,0.5,0.02],
            'green_cube_start': [-0.25,0.5,0.02],

            'red_cube_goal':[0,0.375,0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[.25,.5,0.02],
             'green_cube_goal':[0,.5,0.02]}

edge_end_dict = {
            'red_cube_start':[0.25,0.25,0.02],
            'blue_cube_start': [-0.25,0.25,0.02],
            'yellow_cube_start': [0.25,0.5,0.02],
            'green_cube_start': [-0.25,0.5,0.02],

            'red_cube_goal':[-0.15, 0.1, 0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[.25,.5,0.02],
             'green_cube_goal':[0,.5,0.02]}
env.reset(task_dictionary=edge_start_dict)

# correct plan(s)
print('Edge Start Manipulation Check')
correct = 0
for i in range(8):
    llm_plan = '\nmove_x(-0.15)\nmove_y(-0.15)\npick_block()\nmove_x(0.15)\nmove_y(0.275)\nplace_block()'
    #llm_plan = 'move_z(1.5)\nmove_z(-1.5)\nmove_z(1.5)'
    reward = env.execute_llm_plan(llm_plan)
    correct += reward
    #print('Correct Sequence Reward:', reward)
    env.reset(task_dictionary=edge_start_dict)
print('Edge Start Correct:',correct,'/ 8')

print('Edge End Manipulation Check')
env.reset(task_dictionary=edge_end_dict)
correct = 0
for i in range(8):
    llm_plan = '\nmove_x(0.25)\npick_block()\nmove_x(-0.4)\nmove_y(-0.15)\nplace_block()'
    #llm_plan = 'move_z(1.5)\nmove_z(-1.5)\nmove_z(1.5)'
    reward = env.execute_llm_plan(llm_plan)
    correct += reward
    #print('Correct Sequence Reward:', reward)
    env.reset(task_dictionary=edge_end_dict)
print('Edge End Correct:',correct,'/ 8')
