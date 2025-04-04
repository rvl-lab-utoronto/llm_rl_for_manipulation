"""

This is just some code that James Ross sent for controlling the franka in gensis w/EE control

thanks to him
"""
from pathlib import Path
import traceback
import imageio
import numpy as np
import genesis as gs
import zarr
import numpy as np
from pathlib import Path
from util import Capture, GenerationSettings, generate_dome_points, split_train_validation
from numcodecs import Blosc

settings = GenerationSettings()

gs.init(backend=gs.gpu)

cam_positions = generate_dome_points(settings.radius, settings.azimuth_gridsize, settings.elevation_gridsize)
train_xyz, val_xyz = split_train_validation(cam_positions, settings.n_cams)

# Test
scene = gs.Scene(
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


def create_cam(pos):
    return scene.add_camera(
        res=settings.frame_dims,
        pos=pos,
        lookat=(0., 0., 0.25),
        fov=75,
        GUI=False
    )

train_cameras = [
    create_cam(pos) for pos in train_xyz
]

val_cameras = [
    create_cam(val_xyz[i])
    for i in range(len(val_xyz))
]

base_out = settings.out_dir

ground = scene.add_entity(
    gs.morphs.Box(upper=(2, 2, 0), lower=(-2, -2, -0.5), fixed=True, visualization=False, collision=True)
)

room = scene.add_entity(
    gs.morphs.Mesh(
        file="/workspace/resources/assets/van-gogh.glb",
        fixed=True,
        pos=(-9., 0., 0),
        scale=2.5,
        visualization=True,
        collision=False
    )   
)

cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.45, 0.0, 0.02),
    ),
    surface=gs.surfaces.Default(
        color=(0.8, 0.2, 0.2, 1.0)
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
)
########################## build ##########################
scene.build()

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

franka.set_dofs_kp(kp_values)
franka.set_dofs_kv(kv_values)

franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

def export_state_action(state_action_store: zarr.Group, robot, frame_idx: int):
    # Get robot state and convert to numpy
    qpos = robot.get_dofs_position().cpu().numpy()
    qvel = robot.get_dofs_velocity().cpu().numpy()
    ee_state = robot.get_link('hand')
    current_force = robot.get_dofs_force().cpu().numpy()
    ee_pos = ee_state.get_pos().cpu().numpy()

    ee_quat = ee_state.get_quat().cpu().numpy()

    # Get box height and calculate success
    box_pos = cube.get_pos().cpu().numpy()
    SUCCESS_HEIGHT = 0.1  # 10cm above initial height
    is_success = float(box_pos[2] > (0.02 + SUCCESS_HEIGHT))  # 0.02 is initial box height

    # Prepare data dict
    data = {
        'qpos': qpos,
        'qvel': qvel,
        'ee_pos': ee_pos,
        'ee_rot': ee_quat,
        'action': current_force,
        'timestamp': frame_idx,
        'success': is_success,
        'cube_pos': box_pos  # Added cube position to the data dictionary
    }

    # Check if state_action group exists, if not create it
    if 'state_action' not in state_action_store:
        state_action = state_action_store.create_group('state_action')
        # Create arrays with initial size
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        for key, value in data.items():
            if np.isscalar(value):
                state_action.create_dataset(
                    name=key,
                    shape=(0,),
                    chunks=(10000,),
                    dtype=np.array(value).dtype,
                    compressor=compressor
                )
            else:
                state_action.create_dataset(
                    name=key,
                    shape=(0, *value.shape),
                    chunks=(10000, *value.shape),
                    dtype=value.dtype,
                    compressor=compressor
                )

    state_action = state_action_store['state_action']
    current_len = state_action['qpos'].shape[0]
    
    # Resize all arrays to accommodate new data
    for key, value in data.items():
        current_array = state_action[key]
        new_shape = list(current_array.shape)
        new_shape[0] = current_len + 1
        current_array.resize(tuple(new_shape))
        current_array[current_len] = value


def construct_captures(cams, timestep_capacity: int, cam_paths: list[Path]):
    captures = []
    for camera, cam_path in zip(cams, cam_paths):
        captures.append(Capture(timestep_capacity, camera, cam_path, settings.frame_dims[0], settings.frame_dims[1]))
    return captures


def generate_episode(train_cameras, val_cameras, episode_path: Path):
    # Create directory structure for this episode
    episode_path.mkdir(parents=True, exist_ok=True)

    # Create train and val subdirectories for this episode
    train_dir = episode_path / "train"
    val_dir = episode_path / "validation"

    train_cam_paths = []
    val_cam_paths = []

    # Setup train camera directories
    for train_idx in range(len(train_xyz)):
        cam_path = train_dir / f"capture_{train_idx}"
        cam_path.mkdir(parents=True, exist_ok=True)
        train_cam_paths.append(cam_path)

    # Setup validation camera directories
    for val_idx in range(len(val_xyz)):
        cam_path = val_dir / f"capture_{val_idx}"
        cam_path.mkdir(parents=True, exist_ok=True)
        val_cam_paths.append(cam_path)

    train_dir = episode_path / "train"
    val_dir = episode_path / "validation"
    print("TRAIN XYZ", train_xyz)

    train_captures = construct_captures(train_cameras, 600, train_cam_paths)
    val_captures = construct_captures(val_cameras, 4, val_cam_paths)

    episode_store = zarr.DirectoryStore(str(episode_path / "episode.zarr"))
    episode_root = zarr.group(store=episode_store)

    train_intrinsics = [cam.intrinsics for cam in train_captures]
    train_extrinsics = [cam.extrinsics for cam in train_captures]
    val_intrinsics = [cam.intrinsics for cam in val_captures]
    val_extrinsics = [cam.extrinsics for cam in val_captures]


    np.save(train_dir / "intrinsics.npy", np.stack(train_intrinsics))
    np.save(train_dir / "extrinsics.npy", np.stack(train_extrinsics))

    step_count = 0
    render_step_count = 0
    val_render_step_count = 0
    np.save(val_dir / "intrinsics.npy", np.stack(val_intrinsics))
    np.save(val_dir / "extrinsics.npy", np.stack(val_extrinsics))

    def step_sim(n_steps: int):
        nonlocal step_count
        nonlocal render_step_count
        nonlocal val_render_step_count
        for _ in range(n_steps):
            scene.step()

            if step_count % 170 == 0:
                print("VAL RENDER STEP COUNT", val_render_step_count)
                for val_cap in val_captures:
                    val_cap.add_render(val_render_step_count)
                val_render_step_count += 1

            if step_count % 2 == 0: 
                for train_capture in train_captures:
                    train_capture.add_render(step_count)
                export_state_action(episode_root, franka, render_step_count)
                render_step_count += 1
            step_count += 1


    end_effector = franka.get_link('hand')
    pre_grasp_offset = np.array([0.02, 0.0, 0.23])
    grasp_offset = np.array([0.02, 0.0, 0.13])
    lift_offset = np.array([0.02, 0.0, 0.4])

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.45, 0.0, 0.35]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04
    init_path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=200,
    )

    for waypoint in init_path:  
        franka.control_dofs_position(waypoint)
        scene.step()

    initial_box_pos = cube.get_pos().cpu().numpy()

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=initial_box_pos + pre_grasp_offset,
        quat=np.array([0, 1, 0, 0]),
    )
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=100, 
    )
    # execute the planned path
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        step_sim(1)

    step_sim(100)

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=initial_box_pos + grasp_offset,
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    step_sim(100)

    # grasp
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-0.7, -0.7]), fingers_dof)

    step_sim(100)

    # lift (box initial position + 1m in z)
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=initial_box_pos + lift_offset,
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)

    step_sim(200)

    for i, train_cam in enumerate(train_captures):
        video_path = train_cam_paths[i] / "rgb.mp4"
        train_cam.camera.stop_recording(save_to_filename=str(video_path), fps=30)
    
    for i, val_cam in enumerate(val_captures):
        video_path = val_cam_paths[i] / "rgb.mp4"
        val_cam.camera.stop_recording(save_to_filename=str(video_path), fps=30) 


def reset_robot():
    """Reset the robot to its initial state"""
    # Reset joint positions to initial configuration (all zeros for Franka)
    initial_qpos = np.zeros(franka.n_dofs)
    # Set gripper to open position
    initial_qpos[-2:] = 0.04

    # Hard reset position and ensure velocity is zeroed
    franka.set_dofs_position(initial_qpos, zero_velocity=True)
    franka.zero_all_dofs_velocity()


# Main execution
for i in range(settings.num_episodes):
    episode_path = base_out / f"episode_{str(i).zfill(3)}"
    try:
        scene.reset()
        reset_robot()

        # Randomize cube position with ±0.1m offset on x and y
        random_offset_x = np.random.uniform(-0.1, 0.1)
        random_offset_y = np.random.uniform(-0.1, 0.1)
        cube.set_pos((0.45 + random_offset_x, 0.0 + random_offset_y, 0.02))

        generate_episode(train_cameras, val_cameras, episode_path)
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to generate episode {i}: {e}")
        break
