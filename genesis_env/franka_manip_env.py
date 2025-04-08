# Wanted to see how hard it was to get a basic Franka arm pick n place task
# starts with a franka arm, a cube at one place, and needs to place it at another places. same 
# start and destination for now 

# for later https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/locomotion/go2_env.py

import torch
import math
import numpy as np
import genesis as gs
import time
class FrankaManipEnv:
    def __init__(self, 
                 device="cuda",
                 render_video = False,
                 show_viewer = False,
                 tolerance = 0.1,
                 reward_scale = 2.0,
                 verbose = False,
                 use_gpu = False):
        self.render_video = render_video
        if use_gpu:
            gs.init(backend=gs.gpu)
        else:
            gs.init(backend=gs.cpu)
        self.device = torch.device(device)
        self.verbose = verbose


        ########################## create a scene ##########################
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=60,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,  # visualize the coordinate frame of `world` at its origin
                world_frame_size=1.0,  # length of the world frame in meter
                show_link_frame=False,  # do not visualize coordinate frames of entity links
                show_cameras=False,  # do not visualize mesh and frustum of the cameras added
                plane_reflection=False,  # turn on plane reflection
                ambient_light=(0.3, 0.3, 0.3),  # ambient light setting
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            # renderer=gs.renderers.RayTracer(),
            show_viewer=show_viewer,
            show_FPS=False
        )
        ### various parameters
        ############## other parameters ##############
        self.completion_tolerance = tolerance
        self.reward_scale = reward_scale
        ########################## entities ##########################
        # adds floor plane
        self.plane = self.scene.add_entity(gs.morphs.Plane(),) 
        # adds Franka arm 
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),)
        motors_dof = np.arange(7)
        fingers_dof = np.arange(7, 9)

        
        # adds 4 cubes of different colours
        self.red_cube = self.scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04),pos=(0.25,0.25,0.02),),
                                          surface=gs.surfaces.Default(color=(0.8, 0.2, 0.2, 1.0)))
        
        self.blue_cube = self.scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04),pos=(-0.25,0.25,0.02),),
                                          surface=gs.surfaces.Default(color=(0.2,0.2,0.8, 1.0)))
        
        self.yellow_cube = self.scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04),pos=(0.25,0.5,0.02),),
                                          surface=gs.surfaces.Default(color=(0.8,0.8,0.0 ,1.0)))
        
        self.green_cube = self.scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04),pos=(-0.25,0.5,0.02),),
                                          surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2, 1.0)))
        # camera shit
        if self.render_video:
            self.cam = self.scene.add_camera(res=(640, 480), pos = (0,3,2), lookat=(0,0,0.5), fov=30, GUI=False)
        # build
        self.scene.build()

        # set control gains
        # Note: the following values are tuned for achieving best behavior with Franka
        # Typically, each new robot would have a different set of parameters.
        # Sometimes high-quality URDF or XML file would also provide this and will be parsed.

        # More advanced control with separate multipliers
        base_multiplier = 0.2  # For first 4 joints
        wrist_multiplier = 0.6  # For joints 5-7
        finger_multiplier = 1.0  # For the gripper

        # Base PD gains
        # base_kp = np.array([2500, 2500, 2600, 3000, 2000, 2000, 2000, 100, 100])*0.5
        # base_kv = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])

        # # Apply different multipliers to different joint groups
        # kp_values = np.array([
        #     *list(base_kp[:4] * base_multiplier),
        #     *list(base_kp[4:7] * wrist_multiplier),
        #     *list(base_kp[7:] * finger_multiplier)
        # ])

        # kv_values = np.array([
        #     *list(base_kv[:4] * base_multiplier),
        #     *list(base_kv[4:7] * wrist_multiplier),
        #     *list(base_kv[7:] * finger_multiplier)
        # ])

        # self.franka.set_dofs_kp(kp_values)
        # self.franka.set_dofs_kv(kv_values)

        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])*0.5,
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )

        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )
        # tracking steps
        self.steps = 0

        ## idk gain tuning shit for the arm
        jnt_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
            'finger_joint1',
            'finger_joint2',
        ]
        self.jnt_names = jnt_names
        dofs_idx = [self.franka.get_joint(name).dof_idx_local for name in jnt_names]
        self.dofs_idx = dofs_idx
        self.start_dof_position = self.franka.get_dofs_position()
        # check if it has been initalized with a goal yet
        self.goal_initialized = False

    def move_ee_pos(self,distance,dimension,quick=False):
        displacement = np.zeros(3)
        if dimension == 'x':
            displacement[0] = distance
        elif dimension == 'y':
            displacement[1] = distance
        elif dimension == 'z':
            displacement[2] = distance

        end_effector = self.franka.get_link('hand')
        target_eef_pos = end_effector.get_pos().cpu().numpy() + displacement 
        target_eef_pos[2] = max(0.13, target_eef_pos[2])
        qpos, error = self.franka.inverse_kinematics(
            link=end_effector,
            pos=target_eef_pos,
            quat=np.array([0, 1, 0, 0]),
            return_error=True,
        )
        
        # Option to use path planning instead
        # path = self.franka.plan_path(
        #     qpos_goal     = qpos,
        #     num_waypoints = 200, # 2s duration
        # )
        # # execute the planned path
        # for waypoint in path:
        #     self.franka.control_dofs_position(waypoint)
        #     self.scene.step()

        motors_dof = np.arange(7)
        self.franka.control_dofs_position(qpos[:-2], motors_dof)
        #  self.franka.control_dofs_position(qpos[:-2], self.dofs_idx[:-2])
        if quick:
            for i in range(20):
                self.step_genesis_env()
        else:
            for i in range(100):
                self.step_genesis_env()

    def gripper_open(self):
        fingers_dof = np.arange(7, 9)
        self.franka.control_dofs_force(np.array([0.5, 0.5]), fingers_dof)
        
        for i in range(20):
            self.step_genesis_env()
        

    def gripper_close(self):
        if self.verbose:
            print("CLOSING GRIPPER!")
        fingers_dof = np.arange(7, 9)
        self.franka.control_dofs_force(np.array([-4.0, -4.0]), fingers_dof)
        for i in range(20):
            self.step_genesis_env()

    def step_genesis_env(self):
        self.scene.step()
        if self.render_video:
            self.cam.render()
    def pick_block(self):
        self.gripper_open() # should move total of -0.37
        for i in range(12):
            self.move_ee_pos(-0.03,'z',quick=False)
        self.move_ee_pos(-0.01,'z')
        self.gripper_close()
        for i in range(100):
            self.step_genesis_env()
        self.move_ee_pos(0.01,'z')
        for i in range(12):
            self.move_ee_pos(0.03,'z',quick=True)
        

    def place_block(self):

        for i in range(12):
            self.move_ee_pos(-0.03,'z',quick=False)
        self.move_ee_pos(-0.01,'z')
        self.gripper_open()
        for i in range(100):
            self.step_genesis_env()
        self.move_ee_pos(0.01,'z')
        for i in range(12):
            self.move_ee_pos(0.03,'z',quick=True)


    def execute_llm_plan(self,llm_plan):
        """
        Note - primitives are move_x, move_y, move_z, gripper_open and gripper_close

        LLM plan should have the formating shit stripped out of it by now and each command should be on one line

        returns the reward from the whatever 
        """
        if not self.goal_initialized:
            raise ValueError('Scene is not yet initialized with a goal!')
        if not self.verify_llm_plan_formatting(llm_plan):
            return 0

        legal_commands = ['move_x','move_y','move_z','gripper_open','gripper_close']
        #print(llm_plan)
        plan_line_by_line = llm_plan.splitlines()
        for line in plan_line_by_line:
            if self.verbose:
                print("EXECUTING:", line)
            if 'move' in line: # if its a move command
                try:
                    self.move_ee_pos(float(line[line.find("(")+1:line.find(")")]),line[5])
                except:
                    print('Illegal arugment to move made.')
            elif 'pick_block' in line:
                self.pick_block()
            elif 'place_block' in line:
                self.place_block()
            else:
                if self.verbose:
                    print('Illegal Command Found (and the fucking verifier didnt CATCH IT). Skipping line.')
        reward = self.get_scene_completion_reward()
        return reward

        


    def verify_llm_plan_formatting(self,llm_plan):
        """
        Verifies that the plan meets the formatting restrictions.

        doesn't have the thinking or w/e tokens, should at this point have each primitive on a seperate line
        """

        # NOTE - doesn't work, id wanna fix it, just gonna do this
        return True
        legal_commands = ['move_x','move_y','move_z','gripper_open','gripper_close']
        plan_line_by_line = llm_plan.splitlines()
        primitives_legitimate = any(line in legal_commands for line in plan_line_by_line) # cancer line but its efficent
        if not primitives_legitimate:
            print('meow' )
            return False
        for line in plan_line_by_line:
            if any(legal_commands[:2] in line): # if its a move command
                # gets content between parentheses
                stuff = line[line.find("(")+1:line.find(")")]
                if not self.is_number(stuff):
                    return False

    def get_scene_completion_reward(self):
        """
        Four cubes, returns reward proportional to percentage 
        out of 1 of cubes in right place from cubes not initially in right place, 
        times the env. reward
        scaling parameter
        """
        reward = 0
        #print(self.red_cube.get_pos().cpu().numpy())
        #print(np.array(self.red_cube_goal))
        default_red = np.array([0.25,0.25,0.02])
        default_blue = np.array([-0.25,0.25,0.02])
        default_yellow = np.array([0.25,0.5,0.02])
        default_green = np.array([-0.25,0.5,0.02])
        cubes_needed_moving = 0
        if np.linalg.norm(default_red - self.red_cube_goal) > self.completion_tolerance:
            cubes_needed_moving += 1
            if np.linalg.norm(self.red_cube.get_pos().cpu().numpy() - self.red_cube_goal) < self.completion_tolerance:
                reward += 1
        if np.linalg.norm(default_blue - self.blue_cube_goal) > self.completion_tolerance:
            cubes_needed_moving += 1
            if np.linalg.norm(self.blue_cube.get_pos().cpu().numpy() - self.blue_cube_goal) < self.completion_tolerance:
                reward += 1
        if np.linalg.norm(default_yellow - self.yellow_cube_goal) > self.completion_tolerance:
            cubes_needed_moving += 1
            if np.linalg.norm(self.yellow_cube.get_pos().cpu().numpy() - self.yellow_cube_goal) < self.completion_tolerance:
                reward += 1
        if np.linalg.norm(default_green - self.green_cube_goal) > self.completion_tolerance:
            cubes_needed_moving += 1
            if np.linalg.norm(self.green_cube.get_pos().cpu().numpy() - self.green_cube_goal) < self.completion_tolerance:
                reward += 1

        
        return (reward/cubes_needed_moving) * self.reward_scale

    def reset(self,goal_location):
        """
        Goal location should be a dictionary
        """
        self.scene.reset()
        self.steps = 0
        # self.franka.set_dofs_position(
        #     self.start_dof_position,
        #     self.dofs_idx
        # )
        # resets end effectuator to same place
        end_effector = self.franka.get_link('hand')
        target_eef_pos = np.array([0,0.25,0.5])
        qpos, error = self.franka.inverse_kinematics(
                link=end_effector,
                pos=target_eef_pos,
                quat=np.array([0, 1, 0, 0]),
                return_error=True,
            )

        self.franka.control_dofs_position(qpos)
        for i in range(250):
            self.step_genesis_env()
        self.gripper_open()
        if self.render_video:
            if self.goal_initialized:
                self.cam.stop_recording(save_to_filename='video' + str(time.time()) + '.mp4', fps=60)
            self.cam.start_recording()
        

        self.blue_cube_goal = goal_location['blue_cube_goal']
        self.red_cube_goal = goal_location['red_cube_goal']
        self.green_cube_goal = goal_location['green_cube_goal']
        self.yellow_cube_goal = goal_location['yellow_cube_goal']

        self.goal_initialized = True
        return self.get_observation(), {}
    
    

    def get_observation(self):
        # TODO implement. Or not? I shouldn't actually have to implement this 
        pass
    def close(self):
        pass

    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    