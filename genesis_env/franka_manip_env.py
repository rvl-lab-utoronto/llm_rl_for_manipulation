# Wanted to see how hard it was to get a basic Franka arm pick n place task
# starts with a franka arm, a cube at one place, and needs to place it at another places. same 
# start and destination for now 

# for later https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/locomotion/go2_env.py

import torch
import math
import numpy as np
import genesis as gs
import gymnasium as gym
class FrankaManipEnv:
    def __init__(self, 
                 num_envs = 20,
                 timeout = 250,
                 device="cuda",
                 render_video = False):
        self.render_video = render_video
        gs.init(backend=gs.gpu)
        self.device = torch.device(device)

        self.num_envs = num_envs

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
            show_viewer=False,
            show_FPS=True
        )
        ### various parameters
        ############## other parameters ##############
        self.destination = torch.tensor((0.25, -0.25, 0.02),device=self.device)
        self.cube_start = (0.25, 0.25, 0.02)
        self.state_size = 48
        self.action_size = 7
        # gymnasium stuff 
        self.single_observation_space = gym.spaces.Box(-np.inf,np.inf,shape=(self.state_size,),dtype=float)
        self.single_action_space = gym.spaces.Box(-1.0,1.0,shape=(self.action_size,),dtype=float)
        self.completion_tolerance = torch.ones(self.num_envs,device = self.device) * 0.01
        ########################## entities ##########################
        # adds floor plane
        self.plane = self.scene.add_entity(gs.morphs.Plane(),) 
        # adds Franka arm 
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),)
        motors_dof = np.arange(7)
        fingers_dof = np.arange(7, 9)

        # set control gains
        # Note: the following values are tuned for achieving best behavior with Franka
        # Typically, each new robot would have a different set of parameters.
        # Sometimes high-quality URDF or XML file would also provide this and will be parsed.

        # More advanced control with separate multipliers
        base_multiplier = 0.2  # For first 4 joints
        wrist_multiplier = 0.6  # For joints 5-7
        finger_multiplier = 1.0  # For the gripper

        # Base PD gains
        base_kp = np.array([2500, 2500, 2600, 3000, 2000, 2000, 2000, 100, 100])
        base_kv = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])

        # Apply different multipliers to different joint groups
        kp_values = np.array([
            *list(base_kp[:4] * base_multiplier),
            *list(base_kp[4:7] * wrist_multiplier),
            *list(base_kp[7:] * finger_multiplier)
        ])

        kv_values = np.array([
            *list(base_kv[:4] * base_multiplier),
            *list(base_kv[4:7] * wrist_multiplier),
            *list(base_kv[7:] * finger_multiplier)
        ])

        self.franka.set_dofs_kp(kp_values)
        self.franka.set_dofs_kv(kv_values)

        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )
        # adds cube 
        self.cube = self.scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04),pos=self.cube_start,),
                                          surface=gs.surfaces.Default(color=(0.8, 0.2, 0.2, 1.0)))
        # adds camera
        if self.render_video:
            self.cam = self.scene.add_camera(
                res    = (640, 480),
                pos    = (3.5, 0.0, 2.5),
                lookat = (0, 0, 0.5),
                fov    = 30,
                GUI    = False,
            )
        # build
        self.scene.build(n_envs=num_envs, env_spacing=(2.0, 2.0))

        # tracking steps
        self.steps = 0
        self.timeout = timeout

        

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
        

    def step(self, actions):
        # actions should be size (n_envs, 7)

        self.franka.control_dofs_force(
            # clips to [-1,1] range, then normalizes to constraint max/min
            torch.clip(actions,min=-1.0,max=1.0) * self.action_norm, 
            self.dofs_idx[:-2])
        self.scene.step() # steps through the simulator
        if self.render_video:
            self.cam.render()
        obs,reward = self.get_observation(),self.get_reward()
        self.steps += 1
        done = self.steps > self.timeout
        if done:
            self.reset()
            dones = self.yes_done
        else:
            dones = self.no_done
        
        return obs, reward, dones, self.no_done, {}

    def reset(self,seed = None,options = {}):

        # resets franka arm
        #end_effector = self.franka.get_link('hand')
        #end_effector.set_pos(self.ee_start_pos)
        # resets cube
        #self.cube.set_pos((self.cube_start))
        #print(self.cube.get_pos())
        self.scene.reset()
        self.steps = 0
        if self.render_video:
            self.cam.stop_recording(save_to_filename='video.mp4', fps=60)
            self.cam.start_recording()
        return self.get_observation(), {}
    
    def get_reward(self):
        cube_pos = self.cube.get_pos() # position of cube (n_env, 3)
        ee_pos = self.franka.get_link('hand').get_pos() # position of hand (n_env, 3)

        # encourages cube to get pushed towards the goal
        cube_goal_distance = torch.linalg.vector_norm(cube_pos-self.destination,dim=1)
        cube_goal_distance_reward = -cube_goal_distance

        # encourages end effector to be near the cube
        ee_cube_distance = torch.linalg.vector_norm(cube_pos-ee_pos,dim=1)
        ee_cube_distance_reward = -ee_cube_distance

        # encourages end effector to put cube between itself and the goal
        # cosine sim. between cube2goal and ee2cube should be large ideally
        cube2goal = (self.destination - cube_pos)
        ee2cube = (cube_pos - ee_pos)
        cos_sim = torch.nn.CosineSimilarity(dim=1) 
        cos_reward = ((cos_sim(cube2goal, ee2cube)-1)/2) # normalizes to [-1,0] range

        # aggregates+weights reward components and normalizes 
        """ reward = (cube_goal_distance_reward + 
                  cos_reward * cube_goal_distance / 3 +
                  cos_reward / 3 +
                  ee_cube_distance_reward / 3 + 
                  cos_reward * ee_cube_distance / 3
                  ) / 1000 # to roughly normalize advantage """
        
        reward = (cube_goal_distance_reward * 2 +
                 cos_reward / 4 +
                 ee_cube_distance_reward / 4
                 ) / 100
        #reward = cube_goal_distance_reward / 1000
        #return reward * (not cube_goal_distance < self.completion_tolerance).long()
        return reward * torch.gt(cube_goal_distance,self.completion_tolerance).long()
        # gives 0 reward if goal is complete 
        if cube_goal_distance < self.completion_tolerance:
            return torch.zeros_like(reward)
        else:
            #return np.float64(reward.item())
            return reward

    def get_observation(self):
        obs = []

        # gets the raw things
        cube_pos = self.cube.get_pos() # position of cube (n_env, 3)
        cube_vel = self.cube.get_vel() # velocity of cube (n_env, 3)
        end_effector = self.franka.get_link('hand')
        ee_pos = end_effector.get_pos() # position of ee (n_env, 3)
        ee_vel = end_effector.get_vel() # velocity of ee (n_env, 3)

        obs.append(cube_pos)
        obs.append(cube_vel)
        obs.append(ee_pos)
        obs.append(ee_vel)
        # stuff
        jnt_names = self.jnt_names
        joints = [self.franka.get_joint(name) for name in jnt_names]
        for joint in joints:
            obs.append(joint.get_pos()) # position of joint (n_env, 3)
            #obs.append(joint.get_vel()) # position of joint (n_env, 3)
        #self.franka.get_dofs_force(self.dofs_idx)
        obs.append(self.franka.get_dofs_force(self.dofs_idx))
        obs = torch.cat(obs,dim = 1) # obs (n_env,12)
        #print(obs.shape)
        return obs
    def close(self):
        pass
    