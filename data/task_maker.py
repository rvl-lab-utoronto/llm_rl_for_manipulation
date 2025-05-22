import json 
import numpy as np 
import pandas as pd 
"""
pseudo code idea:


1) spawn block locations

2) choose block for manipulation task

3) choose task

    3.1) between true, relative or between
    3.2.1) if true then pick random location and give out location
    3.2.2) if relative then pick other bock, random x, y numbers and output x and y relative to random block
           until that x,y location is in the grid and not on another block
    3.2.3) choose two other blocks at least 2.0 away from each other and specify to put it between the two start locations 


4) for each block in the prompt input their description alias
"""


COLOURS = ["red", "blue", "yellow", "green"]


def load_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def generate_target_description(description_type, block_for_task, locations, template):
    block_name_for_task = COLOURS[block_for_task]
    if description_type == 0:
        target_description = f"coloured {block_name_for_task}" 
    elif description_type == 1:
        target_description = f"coloured {np.random.choice(template[block_name_for_task])}"
    else:
        target_description = get_location_description(block_for_task, locations)
    return target_description
        
def get_location_description(block_for_task, locations):
    
    """ 
    checks if we can use leftmost, rightmost, highest, or lowest first
    
    
    if none, generates a relative description i.e. (-1, 1) from green block
    """
    
    
    x = locations[block_for_task][0]
    y = locations[block_for_task][1]
    
    leftmost = np.min(locations[:, 0])
    rightmost = np.max(locations[:, 0])
    highest = np.max(locations[:, 1])
    lowest = np.min(locations[:, 1])
    
    
    other_locs =  np.concatenate((locations[:block_for_task], locations[block_for_task+1:]), axis=0)
    
    if x == leftmost and x < np.min(other_locs[:,0]):
        target_description = f"leftmost"
        
    elif x == rightmost and x > np.max(other_locs[:,0]):
        target_description = f"rightmost"
        
    elif y == highest and y > np.max(other_locs[:,1]):
        target_description = f"highest"
        
    elif y == lowest and y < np.min(other_locs[:,1]):
        target_description = f"lowest"
        
    else:
        
        other_block = np.random.choice([i for i in range(4) if i != block_for_task])
        
        relative_x = x - locations[other_block][0]
        relative_y = y - locations[other_block][1]
        
        relative_name = COLOURS[other_block]
        
        
        target_description = f"({relative_x}, {relative_y}) from the {relative_name} block"     
    return target_description
def gen_locations(grid):
    
    X = np.linspace(grid['x'][0], grid['x'][1], num = int((grid['x'][1] - grid['x'][0])/grid['step']))
    Y = np.linspace(grid['y'][0], grid['y'][1], num = int((grid['y'][1] - grid['y'][0])//grid['step']))
    
    
    grid_locations = np.dstack(np.meshgrid(X, Y)).reshape(-1,2)
    
    
    locations = np.random.choice(np.arange(grid_locations.shape[0]), size=4, replace=False)
    
    starting_location_dict = {
        'red_cube_start' : (*grid_locations[locations[0]], 0.02),
        'blue_cube_start' : (*grid_locations[locations[1]], 0.02),
        'yellow_cube_start' : (*grid_locations[locations[2]], 0.02),
        'green_cube_start' : (*grid_locations[locations[3]], 0.02)
    }
    
    
    loc_arr = grid_locations[locations]
    leftover_locs = np.delete(grid_locations, locations, axis=0)
    
    return starting_location_dict, loc_arr, grid_locations, leftover_locs


def choose_block(avails):
    block = np.random.choice(avails)
    
    return block


def create_task(template_filename):
    
    template = load_file(template_filename)
    
    location_dict, locations, whole_grid, leftover_locs = gen_locations(template['location_grid'])

    avail_blocks = [0,1,2,3]

    block_for_task = np.random.randint(0, 4)
    avail_blocks = [ i for i in avail_blocks if i != block_for_task]
    
    
    task_type_probs = template['goal_p']
    ps = [v for (key, v) in task_type_probs.items()]
    cat = np.random.choice(3, p=ps)
    
    if cat == 0:
        "task is to move to a true location"
        
        new_loc = np.random.choice(leftover_locs.shape[0])
        
        goal_location = leftover_locs[new_loc]
        
        
    elif cat == 1:
        
        relative_to_block = choose_block(avail_blocks)
        
        avail_blocks = [i for i in avail_blocks if i != relative_to_block]
        
        relative_block_loc = locations[relative_to_block]
        
        rel_pos = np.random.randint(-4, 4, size=(2,))
        
        while np.isin(relative_block_loc + rel_pos, locations).any():
            rel_pos = np.random.randint(-4, 4, size=(2,))
        
        goal_location = relative_block_loc + rel_pos
        
        
    elif cat == 2:
        
        middle_blocks = np.random.choice(avail_blocks, size=2, replace=False)
        
        middle_block_locs = locations[middle_blocks]
        
        unaccesible = list(middle_blocks) + [block_for_task]
        leftover = np.delete(locations, unaccesible, axis=0)[0]
        
        tries = 0
        while np.linalg.norm(middle_block_locs[0] - middle_block_locs[1]) < 0.35 or np.linalg.norm((middle_block_locs[0] + middle_block_locs[1])/2 - leftover) < 0.25:
            middle_blocks = np.random.choice(avail_blocks, size=2, replace=False)
            middle_block_locs = locations[middle_blocks]
            unaccesible = list(middle_blocks) + [block_for_task]
            leftover = np.delete(locations, unaccesible, axis=0)[0]
            tries +=1 
            if tries >= 500:
                break 
        
        if tries >= 500:
            new_loc = np.random.choice(leftover_locs.shape[0])
        
            goal_location = leftover_locs[new_loc]
            cat == 0
        else:
            goal_location = (middle_block_locs[0] + middle_block_locs[1])/2
    
    else:
        raise ValueError("Invalid task type")


    # actually craft text describing goal
    
    if cat  == 0:
        
        
        description_ps = [v for (k,v) in template['description_p'].items()]
        description_type = np.random.choice(3, p=description_ps)
        
        target_description = generate_target_description(description_type, block_for_task, locations, template)
    
        task_text = f"Take the block that is {target_description} and place it at {goal_location}"
        
        
    elif cat == 1:
        description_ps = [v for (k,v) in template['description_p'].items()]
        target_description_type = np.random.choice(3, p=description_ps)
        relative_description_type = np.random.choice(3, p=description_ps)
        target_description = generate_target_description(target_description_type, block_for_task, locations, template)
        relative_description = generate_target_description(relative_description_type, relative_to_block, locations, template)
        
        
        task_text = f"Take the block that is {target_description} and place it {rel_pos[0]} horizontally and {rel_pos[1]} vertically relative to the block that is {relative_description}"
    
    
    elif cat == 2:
        description_ps = [v for (k,v) in template['description_p'].items()]
        target_description_type = np.random.choice(3, p=description_ps)
        relative1_description_type = np.random.choice(3, p=description_ps)
        relative2_description_type = np.random.choice(3, p=description_ps)
        target_description = generate_target_description(target_description_type, block_for_task, locations, template)
        relative1_description = generate_target_description(relative1_description_type, middle_blocks[0], locations, template)
        relative2_description = generate_target_description(relative2_description_type, middle_blocks[1], locations, template)
        
        task_text = f"Take the block that is {target_description} and place it between the blocks that are {relative1_description} and {relative2_description}"
            
    else:
        raise ValueError("Invalid task type")
    
    new_dict = location_dict
    
    
    new_dict['Text Question'] = task_text 
    new_dict['block_for_task'] = block_for_task
    new_dict['cat'] = cat
    
    if cat == 0:
        other_blocks=None
    elif cat == 1:
        other_blocks= relative_to_block
    elif cat == 2:
        other_blocks = middle_blocks
    else:
        raise ValueError("Invalid task type")
    new_dict['other_blocks'] = other_blocks
    for j in range(len(COLOURS)):
        
        if j ==block_for_task:
            new_dict[f"{COLOURS[block_for_task]}_cube_goal"] = (*goal_location, 0.02)
        else:
            new_dict[f"{COLOURS[j]}_cube_goal"] = new_dict[f"{COLOURS[j]}_cube_start"]
    return new_dict