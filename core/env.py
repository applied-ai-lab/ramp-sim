import gym
import os
from gym import spaces
import numpy as np
import math
import carb
import torch
from core.utils.osc import OperationalSpaceController
from core.utils.transforms import *
from core.utils.linear_interpolator import LinearInterpolator


class RAMPEnv(gym.Env):
    """
    A class containing the simulation environment for the RAMP task.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 400.0,
        rendering_dt=1.0 / 10.0,
        max_episode_length=840,
        seed=0,
        headless=False,
        enable_livestream=False,
        enable_viewport=False,

    ) -> None:
        """
        Args:
            skip_frame (int, optional): Number of physics steps to skip. Defaults to 1.
            physics_dt (float, optional): Physics timestep. Defaults to 1.0 / 1000.0.
            rendering_dt (float, optional): Rendering timestep. Defaults to 1.0 / 10.0.
            max_episode_length (int, optional): Maximum episode length. Defaults to 840.
            seed (int, optional): Random seed. Defaults to 0.
            headless (bool, optional): Whether to run in headless mode. Defaults to False.
            enable_livestream (bool, optional): Whether to enable livestream. Defaults to False.
            enable_viewport (bool, optional): Whether to enable viewport. Defaults to False.
        Returns:
            None
        """

        from omni.isaac.kit import SimulationApp

        # Code for starting the UI
        experience = ""
        if headless:
            if enable_livestream:
                experience = ""
            elif enable_viewport:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.render.kit'
            else:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'

        self._simulation_app = SimulationApp(
            {"headless": headless}, experience=experience)
        carb.settings.get_settings().set(
            "/persistent/omnihydra/useSceneGraphInstancing", True)
        self._render = not headless or enable_livestream or enable_viewport
        self.sim_frame_count = 0

        if enable_livestream:
            from omni.isaac.core.utils.extensions import enable_extension

            self._simulation_app.set_setting("/app/livestream/enabled", True)
            self._simulation_app.set_setting("/app/window/drawMouse", True)
            self._simulation_app.set_setting("/app/livestream/proto", "ws")
            self._simulation_app.set_setting(
                "/app/livestream/websocket/framerate_limit", 120)
            self._simulation_app.set_setting("/ngx/enabled", False)
            enable_extension("omni.kit.livestream.native")
            enable_extension("omni.services.streaming.manager")

        # Global variables
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)

        import omni
        import omni.replicator.core as rep
        from omni.isaac.franka import Franka
        from omni.isaac.core.prims.rigid_prim import RigidPrim
        from omni.isaac.core import World

        # Import the USD file that contains the system setup
        world_usd_file = os.getcwd() + "/assets/scene2.usd"
        omni.usd.get_context().open_stage(world_usd_file)
        self._my_world = World(
            physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        # self._my_world.reset()
        self._my_world.clear_instance()
        self._my_world.get_physics_context()._create_new_physics_scene("/World/physicsScene")
        self._my_world.get_physics_context().enable_gpu_dynamics(True)
        omni.timeline.get_timeline_interface().play()

        # Get birds eye camera
        birds_eye_camera_prim_path = "/World/Camera"
        rp1 = rep.create.render_product(
            birds_eye_camera_prim_path, resolution=(1280, 800))
        # Get wrist camera
        wrist_camera_prim_path = "/World/franka/panda_hand/realsense/realsense/realsense_camera"
        rp2 = rep.create.render_product(
            wrist_camera_prim_path, resolution=(1280, 800))
        # Extract the camera data and make it render rgb data
        self.rgb1 = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb2 = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb1.attach([rp1])
        self.rgb2.attach([rp2])

        perspective_camera__path = "/World/perspective_camera"
        rp3 = rep.create.render_product(
            perspective_camera__path, resolution=(1280, 800))
        self.rgb3 = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb3.attach([rp3])
        self.image_frame = 0

        # Robot setup
        self.frank = Franka("/World/franka", "franka")
        self.ee_rigid = RigidPrim(
            prim_path="/World/franka/ee",
            name="ee",
        )
        self.end_effector_name = "ee"
        self.lula_end_effector_name = "right_gripper"

        # Get Beams
        beam_1_prim_path = "/World/Beam1"
        beam_2_prim_path = "/World/Beam2"
        beam_3_prim_path = "/World/Beam3"
        beam_4_prim_path = "/World/Beam4"
        beam_5_prim_path = "/World/Beam5"
        beam_6_prim_path = "/World/Beam6"
        beam_7_prim_path = "/World/Beam7"
        beam_8_prim_path = "/World/Beam8"
        beam_9_prim_path = "/World/Beam9"

        beam_1 = RigidPrim(
            prim_path=beam_1_prim_path,
            name="Beam1",
        )
        beam_2 = RigidPrim(
            prim_path=beam_2_prim_path,
            name="Beam2",
        )
        beam_3 = RigidPrim(
            prim_path=beam_3_prim_path,
            name="Beam3",
        )
        beam_4 = RigidPrim(
            prim_path=beam_4_prim_path,
            name="Beam4",
        )
        beam_5 = RigidPrim(
            prim_path=beam_5_prim_path,
            name="Beam5",
        )
        beam_6 = RigidPrim(
            prim_path=beam_6_prim_path,
            name="Beam6",
        )
        beam_7 = RigidPrim(
            prim_path=beam_7_prim_path,
            name="Beam7",
        )
        beam_8 = RigidPrim(
            prim_path=beam_8_prim_path,
            name="Beam8",
        )
        beam_9 = RigidPrim(
            prim_path=beam_9_prim_path,
            name="Beam9",
        )

        self.beam_list = [beam_1, beam_2, beam_3, beam_4,
                          beam_5, beam_6, beam_7, beam_8, beam_9]

        # Get Pegs
        peg_1_prim_path = "/World/Holder1/Assembly_1/Peg"
        peg_2_prim_path = "/World/Holder1/Assembly_1/Peg_01"
        peg_3_prim_path = "/World/Holder1/Assembly_1/Peg_02"
        peg_4_prim_path = "/World/Holder2/Assembly_1/Peg"
        peg_5_prim_path = "/World/Holder2/Assembly_1/Peg_01"
        peg_6_prim_path = "/World/Holder2/Assembly_1/Peg_02"
        peg_7_prim_path = "/World/Holder3/Assembly_1/Peg"
        peg_8_prim_path = "/World/Holder3/Assembly_1/Peg_01"
        peg_9_prim_path = "/World/Holder3/Assembly_1/Peg_02"
        peg_10_prim_path = "/World/Holder4/Assembly_1/Peg"
        peg_11_prim_path = "/World/Holder4/Assembly_1/Peg_01"
        peg_12_prim_path = "/World/Holder4/Assembly_1/Peg_02"

        peg_1 = RigidPrim(
            prim_path=peg_1_prim_path,
            name="Peg",
        )
        peg_2 = RigidPrim(
            prim_path=peg_2_prim_path,
            name="Peg_01",
        )
        peg_3 = RigidPrim(
            prim_path=peg_3_prim_path,
            name="Peg_02",
        )
        peg_4 = RigidPrim(
            prim_path=peg_4_prim_path,
            name="Peg",
        )
        peg_5 = RigidPrim(
            prim_path=peg_5_prim_path,
            name="Peg_01",
        )
        peg_6 = RigidPrim(
            prim_path=peg_6_prim_path,
            name="Peg_02",
        )
        peg_7 = RigidPrim(
            prim_path=peg_7_prim_path,
            name="Peg",
        )
        peg_8 = RigidPrim(
            prim_path=peg_8_prim_path,
            name="Peg",
        )
        peg_9 = RigidPrim(
            prim_path=peg_9_prim_path,
            name="Peg",
        )
        peg_10 = RigidPrim(
            prim_path=peg_10_prim_path,
            name="Peg",
        )
        peg_11 = RigidPrim(
            prim_path=peg_11_prim_path,
            name="Peg_01",
        )
        peg_12 = RigidPrim(
            prim_path=peg_12_prim_path,
            name="Peg_02",
        )

        self.peg_list = [peg_1, peg_2, peg_3, peg_4, peg_5,
                         peg_6, peg_7, peg_8, peg_9, peg_10, peg_11, peg_12]

        self.seed(seed)
        self.reward_range = (float("inf"), float("inf"))
        gym.Env.__init__(self)

        self.reset_counter = 0
        self.marker_num = 0

        return

    def setup_action_space(self,):
        """
        Internal method for setting up functionality required to execute actions.
        Args:
            None
        Returns:
            None
        """
        import omni
        import lula
        from omni.isaac.dynamic_control import _dynamic_control
        from omni.isaac.core.articulations import ArticulationView, Articulation
        from omni.isaac.core.utils.extensions import get_extension_path_from_name
        from omni.isaac.motion_generation.lula import LulaCSpaceTrajectoryGenerator

        # Start the simulation
        omni.timeline.get_timeline_interface().play()

        # If the simulation is just starting, move the Franka to a good start state
        if self._my_world.is_playing():
            self.frank.initialize()
            if self.reset_counter == 0:
                print("Moving the robot to a start state")
                # Set the desired robot joint states
                self.frank.set_joint_positions(np.array(
                    [0.012, -0.56974876, 2.8512668e-08, -2.8104815,
                     4.78208e-06, 3.0304759, 0.6967839, 0.02, 0.02]))
                for step in range(10):
                    # Allow the robot to servo to the desired joint states
                    self._my_world.step(render=True)

        # Create the impedance controller
        self.osc_control = OperationalSpaceController(
            joint_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
            actuator_range=[-10, 10],
            control_delta=False,
            control_ori=True,
            initial_joint=[0., 0., 0., 0., 0., 0., 0., 0., 0.],
            uncouple_pos_ori=True,
            kp=[80, 80, 80, 150, 150, 150],
            damping_ratio=0.8,
            interpolator_pos=LinearInterpolator(
                ndim=3,
                controller_freq=(1/self._dt),
                policy_freq=(1/(self._dt*80)),
                ramp_ratio=0.2
            ),
            interpolator_ori=LinearInterpolator(
                ndim=3,
                controller_freq=(1/self._dt),
                policy_freq=(1/(self._dt*80)),
                ori_interpolate="euler",
                ramp_ratio=0.8,
            )
        )
        self.osc_control.update_initial_joints(
            self.frank.get_joint_positions()[:7])

        # Acuire the dynamic control interface
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.art = self.dc.get_articulation("/World/franka")
        self.dc.wake_up_articulation(self.art)

        # Create an ArticulationView and Articulation object for the Franka
        self.view = ArticulationView("/World/franka", "franka")
        self.view.initialize()
        self.franka_articulation = Articulation("/World/franka")
        self.franka_articulation.initialize()
        self.articulation_controller = self.franka_articulation.get_articulation_controller()

        self.mg_extension_path = get_extension_path_from_name(
            "omni.isaac.motion_generation")
        self.rmp_config_dir = os.path.join(
            self.mg_extension_path, "motion_policy_configs")
        self.robot_description_path = self.rmp_config_dir + \
            "/franka/rmpflow/robot_descriptor.yaml"
        self.urdf_path = self.rmp_config_dir + "/franka/lula_franka_gen.urdf"

        for body_index, body_name in enumerate(self.view.body_names):
            if self.end_effector_name == body_name:
                self.ee_body_index = body_index

        # Create the Lula path planner
        self.lula_robot = lula.load_robot(
            self.robot_description_path, self.urdf_path).kinematics()
        self.conversion_config = lula.TaskSpacePathConversionConfig()
        self.ik_config = lula.CyclicCoordDescentIkConfig()

        # Initialize a LulaCSpaceTrajectoryGenerator object
        self.c_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=self.robot_description_path,
            urdf_path=self.urdf_path,
        )

        # Parameters for the C-space trajectory generator
        # self.c_space_trajectory_generator.set_solver_param("max_segment_iterations", 256)
        # self.c_space_trajectory_generator.set_solver_param("max_aggragate_iterations", 256)
        # self.c_space_trajectory_generator.set_solver_param("convergence_dt", 20.0)
        # self.c_space_trajectory_generator.set_solver_param("max_dilation_iterations", 256)
        # self.c_space_trajectory_generator.set_solver_param("dilation_dt", 0.5)
        # self.c_space_trajectory_generator.set_solver_param("min_time_span", 0.5)
        # TODO: Improve the planner and obey robot joint limits
        self.c_space_trajectory_generator.set_c_space_position_limits(
            np.array([-20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0]),
            np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]))
        self.c_space_trajectory_generator.set_c_space_jerk_limits(
            np.array([50., 50., 50., 50., 50., 50., 50.]))
        # self.c_space_trajectory_generator.set_c_space_acceleration_limits(
        # np.array([7.0, 4.0, 5.0, 6.5, 7.0, 10.0, 10.0]))

        # Determine the indices of the actuated DOFs
        joint_list = ["panda_joint1",
                      "panda_joint2",
                      "panda_joint3",
                      "panda_joint4",
                      "panda_joint5",
                      "panda_joint6",
                      "panda_joint7",
                      "panda_finger_joint1",
                      "panda_finger_joint2"]
        self.actuated_dof_indices = list()
        for joint_name in joint_list:
            self.actuated_dof_indices.append(
                self.view.get_dof_index(joint_name))
        print("Actuated DOF indices: ", self.actuated_dof_indices)
        self.actuated_dof_indices.sort()

    def get_dt(self):
        """
        Returns the timestep of the simulation
        Args:
            None
        Returns:
            float: timestep of the simulation
        """
        return self._dt

    def step(self,):
        """
        Performs a single step of the simulation

        Returns:
            observations (dict): dictionary of observations
        """
        import omni

        omni.timeline.get_timeline_interface().play()

        self.setup_action_space()

        self._my_world.step(render=True)
        observations = self.get_observations()
        self.reset_counter += 1

        # return observations, reward, done, info
        return observations

    def reset(self):
        """
        Resets the simulation to the initial state
        Args:
            None
        Returns:
            observations (dict): dictionary of observations
        """

        import omni

        omni.timeline.get_timeline_interface().stop()
        omni.timeline.get_timeline_interface().play()
        self._my_world.step(render=True)
        self.reset_counter = 0
        self.setup_action_space()

        observations = self.get_observations()

        return observations

    def get_observations(self):
        """
        Returns the observations of the environment as a dictioanry. Observations include: 
        a list of beam poses, a list of peg poses, the robot's cartesian pose, the robot's 
        force/torque sensor reading, the robot's joint positions, the robot's joint efforts, 
        the robot's gripper state, the wrist camera image, and the birds eye camera image.
        Args:
            None
        Returns:
            observations (dict): dictionary of observations
        """

        obs = {
            "beams": [],
            "pegs": [],
            "robot_cartesian_pose": [],
            "force_sensor": [],
            "robot_joint_position": [],
            "robot_joint_effort": [],
            "robot_gripper": [],
            "wrist_camera": [],
            "birds_eye_camera": []
        }

        obs["robot_cartesian_pose"] = np.concatenate(
            (self.frank.end_effector.get_world_pose()[0],
             self.frank.end_effector.get_world_pose()[1]), axis=0)
        obs["robot_joint_position"] = self.frank.get_joint_positions()
        obs["robot_joint_effort"] = self.frank.get_applied_joint_efforts()
        obs["robot_gripper"] = self.frank.gripper.get_joint_positions()

        obs["birds_eye_camera"] = self.rgb1.get_data()
        obs["wrist_camera"] = self.rgb2.get_data()

        obs["force_sensor"] = self.view._physics_view.get_force_sensor_forces()[
            0, 0, :]

        for beam in self.beam_list:
            position, quat = beam.get_world_pose()
            obs["beams"].append(np.concatenate((position, quat), axis=0))
        for peg in self.peg_list:
            position, quat = peg.get_world_pose()
            obs["pegs"].append(np.concatenate((position, quat), axis=0))

        return obs

    def close(self):
        """
        Close the simulation UI
        Args:
            None
        Returns:
            None
        """
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Args:
            seed (int): the seed to use
        Returns:
            [seed]: the list of seeds used in this env's random number generators
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def operational_force_control(self, action, control_delta=False):
        """
        Action function for moving to a pose with impedance control

        Args:
            action (np.array): 1d-array containing a pose (x,y,z,rw,rx,ry,rz)
            control_delta (bool): if True, the pose is interpreted as a delta pose

        Returns:
            dictionary: observations from the environment
        """
        from omni.isaac.dynamic_control import _dynamic_control

        self.osc_control.use_delta = control_delta

        # Change the joint properties - always assume we were doing position
        # control previously
        dof_props = self.dc.get_articulation_dof_properties(self.art)
        for i in range(self.dc.get_articulation_dof_count(self.art)-2):
            dof_props["driveMode"][i] = _dynamic_control.DriveMode.DRIVE_FORCE
            dof_props["stiffness"][i] = 0.0
            dof_props["damping"][i] = 5.0
        self.dc.set_articulation_dof_properties(self.art, dof_props)

        ee_pose = np.concatenate((
            self.ee_rigid.get_world_pose()[0],
            self.ee_rigid.get_world_pose()[1]), axis=0)

        if action.shape[0] < 7:
            action = np.concatenate((action, ee_pose[3:]))

        self.osc_control.update_initial_joints(
            self.frank.get_joint_positions()[:7])

        self.add_marker(action[:])  # remove to prevent marker visualization

        ee_pose = np.concatenate(
            (self.ee_rigid.get_world_pose()[0],
             self.ee_rigid.get_world_pose()[1]), axis=0)
        self.osc_control.set_goal(action=action, ee_pose=ee_pose)

        for step in range(80):  # look at breaking loop if goal is reached

            # The velocity returned by the ee_rigid body is noisy, don't use it
            ee_pose = np.concatenate(
                (self.ee_rigid.get_world_pose()[0],
                 self.ee_rigid.get_world_pose()[1]), axis=0)
            ee_vel = np.concatenate(
                (self.frank.end_effector.get_linear_velocity()[:],
                 self.frank.end_effector.get_angular_velocity()[:]), axis=0)

            robot_actions = self.osc_control.run_controller(
                jacobian=self.view.get_jacobians(
                )[0, self.ee_body_index-1, :, :7],
                position_jacobian=self.view.get_jacobians(
                )[0, self.ee_body_index-1, 0:3, :7],
                orientation_jacobian=self.view.get_jacobians(
                )[0, self.ee_body_index-1, 3:6, :7],
                ee_pose=ee_pose,
                ee_vel=ee_vel,
                joint_pos=self.frank.get_joint_positions()[:7],
                joint_vel=self.frank.get_joint_velocities()[:7],
                mass_matrix=self.view.get_mass_matrices()[0, :7, :7],
                gravity=self.view.get_generalized_gravity_forces()[0, :7]
            )

            robot_actions = np.concatenate(
                (robot_actions, np.array((0, 0))), 0)

            self.frank.set_joint_efforts(robot_actions)

            self._my_world.step(render=True)

        observations = self.get_observations()

        return observations

    def operational_position_control(self, action, control_delta=False):
        """
        Action function for moving lineary through an array of poses

        Args:
            poses (np.array): 2d-array of N poses to move through
            control_delta (bool): if True, the pose is interpreted as a delta pose

        Returns:
            dictionary: observations from the environment
        """
        import lula
        from omni.isaac.motion_generation import ArticulationTrajectory
        from omni.isaac.motion_generation.lula.utils import get_pose3
        from omni.isaac.dynamic_control import _dynamic_control

        # Change the joint properties - always assume we were doing impedance
        # control previously
        dof_props = self.dc.get_articulation_dof_properties(self.art)
        for i in range(self.dc.get_articulation_dof_count(self.art)):
            dof_props["driveMode"][i] = _dynamic_control.DriveMode.DRIVE_ACCELERATION
            dof_props["stiffness"][i] = 60000.0
            dof_props["damping"][i] = 3000.0
        self.dc.set_articulation_dof_properties(self.art, dof_props)

        path_spec = lula.create_composite_path_spec(
            self.frank.get_joint_positions()[0:-2])

        ee_pose = np.concatenate(
            (self.frank.end_effector.get_world_pose()[0],
             self.frank.end_effector.get_world_pose()[1]), axis=0)

        if action.ndim < 2:
            action = action.reshape(1, -1)
        if action.shape[1] < 7:
            action = np.concatenate(
                (action, np.tile(ee_pose[3:], (action.shape[0], 1))), axis=1)
        if control_delta:
            action = action[:, :3] + ee_pose[:3]
            for i in range(0, action.shape[0]):
                action[i, 3:] = ee_pose[3:] * action[i, 3:]
        spec = lula.create_task_space_path_spec(
            get_pose3(action[0, :3], rot_quat=action[0, 3:]))
        for pose in range(1, action.shape[0]):
            pos = action[pose, :3]
            # TODO: Need to add an offset to the desired pose to account
            # for the offset from the gripper for the Lula model
            spec.add_linear_path(get_pose3(pos, rot_quat=action[pose, 3:]))

        path_spec.add_task_space_path_spec(
            spec, lula.CompositePathSpec.TransitionMode.LINEAR_TASK_SPACE)

        c_space_points = lula.convert_composite_path_spec_to_c_space(path_spec,
                                                                     self.lula_robot,
                                                                     self.lula_end_effector_name,
                                                                     self.conversion_config,
                                                                     self.ik_config
                                                                     )
        trajectory = self.c_space_trajectory_generator.compute_c_space_trajectory(
            np.vstack(c_space_points.waypoints()))

        if trajectory is None:
            print("No trajectory could be generated!")
        else:
            articulation_trajectory = ArticulationTrajectory(
                self.franka_articulation, trajectory, self._dt*2)

            # Returns a list of ArticulationAction meant to be executed on subsequent physics steps
            action_sequence = articulation_trajectory.get_action_sequence()

            # Create frames to visualize the task_space targets
            for i in range(action.shape[0]):
                self.add_marker(action[i, :])

            for act in action_sequence:
                self.franka_articulation.apply_action(act)
                self._my_world.step(render=True)

        observations = self.get_observations()

        return observations

    def gripper_position_control(self, width):
        """
        Action function for moving the gripper fingers given a width

        Args:
            width (float): float representing the width of the gripper fingers

        Returns:
            dictionary: observations from the environment
        """
        from omni.isaac.core.utils.types import ArticulationAction

        for step in range(1, 20):

            gripper_positions = self.frank.gripper.get_joint_positions()

            difference = (width - gripper_positions[0]) * step/19

            self.frank.gripper.apply_action(
                ArticulationAction(joint_positions=[
                                   gripper_positions[0]+difference,
                                   gripper_positions[0]+difference])
            )
            self._my_world.step(render=True)

        for step in range(8):
            self._my_world.step(render=True)

        observations = self.get_observations()

        return observations

    def add_marker(self, pose):
        """
        Add a marker to the stage to visualize the target pose. If markers are added
        then they will act as collision object, this is good for debugging but causes
        unwanted collisions.

        Args:
            pose (np.array): 1d-array of length 7 representing the pose of the marker

        Returns:
            None
        """
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.prims import XFormPrim

        if self.marker_num > 5:
            self.marker_num = 0

        # add a marker to the stage to assist with debugging
        # add_reference_to_stage(get_assets_root_path(
        # ) + "/Isaac/Props/UIElements/frame_prim.usd", f"/target_{self.marker_num}")
        # frame = XFormPrim(f"/target_{self.marker_num}", scale=[.01, .01, .01])
        # position = pose[0:3]
        # orientation = pose[3:]
        # frame.set_world_pose(position, orientation)
        # self.marker_num += 1

        return None
