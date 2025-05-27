from franka_manip_env import FrankaManipEnv

env = FrankaManipEnv(render_video=False,show_viewer=False,verbose=False, use_gpu=True) # change however you want
import pandas as pd
import numpy as np

# note - manipulator starts at [0,0.25]
NUM_TRIALS = 1
DATA_PATH = '../data/task_dataset.csv'
EE_START = (0,0.25,0)


default_dict = {
            'red_cube_start':[0.25,0.25,0.02],
            'blue_cube_start': [-0.25,0.25,0.02],
            'yellow_cube_start': [0.25,0.5,0.02],
            'green_cube_start': [-0.25,0.5,0.02],

            'red_cube_goal':[0.25,0.25,0.02],
             'blue_cube_goal':[-.25,.25,0.02],
             'yellow_cube_goal':[0,0.375,0.02],
             'green_cube_goal':[-0.25,0.5,0.02]}
env.reset(task_dictionary=default_dict)

correct = 0
num_tasks = 0
# loads data as pandas dataframe
data = pd.read_csv(DATA_PATH)
for index, row in data.iterrows():
    num_tasks += 1
    for trial in range(NUM_TRIALS):
        # makes reset dictionary, resets environment
        reset_dict = {'red_cube_start':row['red_cube_start'],
                        'blue_cube_start': row['blue_cube_start'],
                        'yellow_cube_start': row['yellow_cube_start'],
                        'green_cube_start': row['green_cube_start'],

                        'red_cube_goal':row['red_cube_goal'],
                        'blue_cube_goal':row['blue_cube_goal'],
                        'yellow_cube_goal':row['yellow_cube_goal'],
                        'green_cube_goal':row['green_cube_goal']}
        env.reset(task_dictionary=reset_dict)

        # gets colour of goal cube 
        goal_block_idx = row['block_for_task']
        goal_block_colour = ''
        if goal_block_idx == 0:
            goal_block_colour = 'red'
        elif goal_block_idx == 1:
            goal_block_colour = 'blue'
        elif goal_block_idx == 2:
            goal_block_colour = 'yellow'
        elif goal_block_idx == 3:
            goal_block_colour = 'green'
        # assigns start and end locations based on that colour
        goal_block_start = row[goal_block_colour + '_cube_start']
        goal_block_end = row[goal_block_colour + '_cube_goal']

        ### next section constructs the llm plan
        llm_plan = ''
        # moves manipulator EE to above the goal cube
        disp = np.array(goal_block_start) - np.array(EE_START)
        llm_plan += '\nmove_x(' + str(disp[0]) + ')' + '\nmove_y(' + str(disp[1]) + ')'
        # picks
        llm_plan += '\npick_block()'
        # moves manipulator EE to above goal cube final location
        disp = np.array(goal_block_end) - np.array(goal_block_start)
        llm_plan += '\nmove_x(' + str(disp[0]) + ')' + '\nmove_y(' + str(disp[1]) + ')'
        # places
        llm_plan += '\nplace_block()'

        # executes plan in environment
        reward = env.execute_llm_plan(llm_plan)
        correct += reward
        #print('Correct Sequence Reward:', reward)
print('Correct:',correct,'/',NUM_TRIALS*num_tasks)
print('Percent:',correct/(NUM_TRIALS*num_tasks))
