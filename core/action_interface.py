import os
import sys
import re
import math
import numpy as np

from core.utils.general_utils import AttrDict
from core.utils.transforms import *
from core.const import BEAMS, PEGS
from planner.beam_assembly.beam_assembly_parser import ElementType


class ActionInterface(object):
    """
    This class provides a high-level interface between the planner and 
    the environment. The class interfaces with the robot to move the 
    robot, grasp beams, grasp pegs, and place beams.
    """

    def __init__(self, fixed_beam_id="b7"):
        """
        Args:
            fixed_beam_id: ID of a beam fixed on a table
        """

        self.observations = None
        self.env = None
        self.useful_orientations()
        self.named_poses()

        self.fixed_beam_id = fixed_beam_id

        self.beams = BEAMS
        self.pegs = PEGS

        self.grasped_beam_id = None
        self.grasped_beam_link_id = None

        self.ee_orientation = self.gripper_down

    def named_poses(self,):
        """
        Define named poses for the robot
        Args:
            None
        Returns:
            None
        """
        self.home_pose = np.concatenate((np.array([0.4, 0.0, 0.3]),
                                         self.gripper_down))

    def useful_orientations(self,):
        """
        Define useful orientations for the robot
        Args:
            None
        Returns:
            None
        """
        # gripper down
        self.gripper_down = euler2mat(np.array([3.14, 0., 3.14]))
        self.gripper_down = w_quat(mat2quat(self.gripper_down))

        # gripper down - +90 deg
        self.gripper_down_90_pos = euler2mat(np.array([3.14, 0.0, 3.14+1.57]))
        self.gripper_down_90_pos = w_quat(mat2quat(self.gripper_down_90_pos))

        # gripper down - -90 deg
        self.gripper_down_90_neg = euler2mat(np.array([3.14, 0.0, 1.57]))
        self.gripper_down_90_neg = w_quat(mat2quat(self.gripper_down_90_neg))

        # gripper down - -180 deg
        self.gripper_down_180 = euler2mat(np.array([3.14, 0.0, 0.0]))
        self.gripper_down_180 = w_quat(mat2quat(self.gripper_down_180))

    def move_to_neutral(self,):
        """
        Move the robot to a neutral pose
        Args:
            None
        Returns:
            None
        """
        obs = self.env.operational_position_control(self.home_pose)
        self.observations = self.transform_beam_pose(obs)

    def pickup_beam(self, target):
        """
        Grasp a beam at a specific link
        Args:
            target (str): A target beam id with link id for grasping. E.g. b7l1
        Returns:
            True
        """
        beam_id, link_id = self.get_beam_link_id(target)

        link_pose = self.get_beam_link_pose(beam_id, link_id)
        pos, ori = mat2pose(link_pose)
        grasped_link_euler = quat2axisangle(ori)

        # use these information to update the approach pose later.
        self.grasped_link_euler = grasped_link_euler
        self.grasped_beam_id = beam_id
        self.grasped_beam_link_id = link_id

        # open the gripper and move above the target link
        self.env.gripper_position_control(0.02)
        poses = np.concatenate((np.array([[pos[0], pos[1], 0.15], [pos[0], pos[1], 0.038],]),
                                np.array([self.gripper_down, self.gripper_down,])), axis=1)
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        # grasp the beam
        self.env.gripper_position_control(0.008)
        print("Grasped!")

        # move upwards
        poses = np.concatenate((np.array([[pos[0], pos[1], 0.15],]),
                                np.array([self.gripper_down,])), axis=1)
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)
        return True

    def move_to_input_area(self):
        """
        Move to the input area.
        Args:
            None
        Returns:
            None
        """
        print("Moving to the input area")
        pose = self.input_origin.copy()
        pose[0] = pose[0] + 0.0
        pose[1] = pose[1] - 0.0
        pose[2] = pose[2] + 0.35
        obs = self.env.operational_position_control(pose)
        self.observations = self.transform_beam_pose(obs)

    def move_to_intermediate_area(self):
        """
        Move to the intermediate area.
        Args:
            None
        Returns:
            None
        """
        print("Moving to the intermediate area")
        poses = self.input_origin.copy()
        poses[0] = poses[0] - 0.0
        poses[1] = poses[1] - 0.2
        poses[2] = poses[2] + 0.35
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)
        self.move_neutral_ori()

    def move_to_assembly_area(self):
        """
        Move to the assembly area.
        Args:
            None
        Returns:
            None
        """
        print("Moving to the assembly area")
        poses = self.input_origin.copy()
        poses[0] = poses[0] - 0.0
        poses[1] = poses[1] - 0.4
        poses[2] = poses[2] + 0.35
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

    def move_beam_to_approach_pose(self, target, insert_end):
        """
        Move the beam to the approach pose for insertion
        Args:
            target (str): A target beam id.
            insert_end (BeamComponent): An instance of BeamComponent class 
                                        containing the insertion endpoint joint.
        Returns:
            None
        """

        print("Moving to the approach pose")
        # Transform the approach pose of the insertion endpoint to the approach poes of the link that the robot grasps.
        self.update_beam_approach_pose(
            self.grasped_beam_id, self.grasped_beam_link_id, insert_end
        )

        self.move_neutral_ori()

        target_pose = self.beams[target]
        trans, quat = mat2pose(target_pose)
        euler = quat2axisangle(quat)

        self.ee_orientation = self.gripper_down

        poses = np.concatenate((np.array([trans[0], trans[1], 0.15]),
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses, control_delta=False)
        self.observations = self.transform_beam_pose(obs)

        # FIXME: rotate a gripper with the difference in orientation of grasped and target beam
        if np.abs(self.grasped_link_euler[2] - euler[2]) > 2:
            print("Rotate gripper by 180 deg")
            self.ee_orientation = self.rotate_gripper(
                self.ee_orientation, np.pi)
        elif np.abs(self.grasped_link_euler[2] - euler[2]) > 1.0:
            print("Rotate gripper by 90 deg")
            self.ee_orientation = self.rotate_gripper(
                self.ee_orientation, np.pi / 2.0)

        # Slowly move the beam to the approach pose.
        poses = np.concatenate((np.array([trans[0], trans[1], 0.038]),
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

    def move_to_beam(self, target):
        """
        Move to the origin of the target beam
        Args:
            target (str): A target beam id
        Returns:
            None
        """
        print("Moving to the beam {}".format(target))
        self.move_neutral_ori()
        beam_id = self.get_beam_id(target)

        beam_id_int = int(beam_id[1:])
        pose = self.observations["beams"][beam_id_int-1]

        poses = np.concatenate((np.array([[pose[0], pose[1], 0.2],]),
                                np.array([self.gripper_down,])), axis=1)
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

    def move_to_peg(self, target):
        """
        Move to the target peg
        Args:
            target (str): A target peg id
        Returns:
            True
        """
        print("Moving to the peg {}".format(target))
        peg_id = re.match("(p\d*)i", target)[1]
        pose = self.observations["pegs"][int(peg_id[1:])-1]
        poses = np.concatenate(([pose[0], pose[1], 0.3],
                                self.gripper_down))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        return True

    def pickup_peg(self, target):
        """
        Grasp a peg
        Args:
            marker_id: the peg's marker id
        Returns:
            True
        """

        print("Pick up the peg {}".format(target))

        self.env.gripper_position_control(0.04)  # gripper open

        # Move down
        pose = self.observations["pegs"][int(target[1:])-1]
        poses = np.concatenate(([pose[0], pose[1], 0.084],
                                self.gripper_down))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        self.env.gripper_position_control(0.0095)  # gripper close

        # Move up
        poses = np.concatenate(([pose[0], pose[1], 0.2],
                                self.gripper_down))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        return True

    def insert_peg(self, bj1, bj2, peg):
        """
        Insert the grasped peg to the hole of bj1 and bj2.
        Args:
            bj1: A beam id with a joint id
            bj2: A beam id with a joint id
            peg: A peg id
        Returns:
            True
        """

        print("Inserting the peg {}...".format(peg))

        # Move above the beam
        beam1_id, beam1_joint_id = self.get_beam_joint_id(bj2)
        pose1 = self.get_beam_joint_pose(beam1_id, beam1_joint_id)
        pose = mat2pose(pose1)[0]
        poses = np.concatenate(([pose[0], pose[1], 0.2],
                                self.gripper_down))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        # Move down into the hole
        poses = np.concatenate(([pose[0], pose[1], 0.1],
                                self.gripper_down))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        return True

    def assemble_beam_square(self, grasped_beam, target_beam_component):
        """
        Assemble beams.
        Args:
            grasped_beam (str): A grasped beam id with a joint id
            target_beam_component (BeamComponent): A grasped beam component
        Returns:
            True
        """
        target_beam = target_beam_component.name

        if target_beam_component.type.value == ElementType.THRU_F.value:
            self.assemble_thru_f(grasped_beam, target_beam)
        elif target_beam_component.type.value == ElementType.IN_F_END.value:
            self.assemble_in_f_end(grasped_beam, target_beam)
        elif target_beam_component.type.value == ElementType.IN_F.value:
            self.assemble_in_f(grasped_beam, target_beam)

        return True

    def assemble_thru_f(self, grasped_beam, target_beam):
        """
        Assembly for thru_f joint type
        Assemble beams.
        Args:
            grasped_beam (str): A grasped beam id with a joint id
            target_beamt (str): A target beam id with a joint id
        Returns:
            None
        """
        print("Assemble thru_f...")
        raise NotImplementedError

    def assemble_in_f(self, grasped_beam, target_beam):
        """
        Assembly for in_f joint type
        Assemble beams.
        Args:
            grasped_beam (str): A grasped beam id with a joint id
            target_beamt (str): A target beam id with a joint id
        Returns:    
            None
        """

        print("Assemble in_f...")
        grasped_beam_id, grasped_joint_id = self.get_beam_joint_id(
            grasped_beam)
        beam_id, joint_id = self.get_beam_joint_id(target_beam)

        joint_pos, joint_ori = mat2pose(
            self.get_beam_joint_pose(beam_id, joint_id))
        grasped_joint_pos, grasped_joint_ori = mat2pose(
            self.get_beam_joint_pose(grasped_beam_id, grasped_joint_id))

        # find the insertion axis
        insert_axis = np.where(
            np.abs(
                self.beams["{}t".format(grasped_beam_id)][:3, 3]
                - self.beams["{}a".format(grasped_beam_id)][:3, 3]
            )[:2]
            > 0.02
        )[0][0]

        if insert_axis == 0:
            current_pos = self.observations["robot_cartesian_pose"][:3]
            poses = np.concatenate(([current_pos[0]-(grasped_joint_pos[0]-joint_pos[0])*0.6, joint_pos[1], 0.04],
                                    self.ee_orientation))
            obs = self.env.operational_position_control(poses)
            self.observations = self.transform_beam_pose(obs)
            current_pos = self.observations["robot_cartesian_pose"][:3]
            poses = np.concatenate(([current_pos[0]-(grasped_joint_pos[0]-joint_pos[0]), joint_pos[1], 0.04],
                                    self.ee_orientation))
            obs = self.env.operational_position_control(poses)
            self.observations = self.transform_beam_pose(obs)
        elif insert_axis == 1:
            current_pos = self.observations["robot_cartesian_pose"][:3]
            poses = np.concatenate(([joint_pos[0], current_pos[1]-(grasped_joint_pos[1]-joint_pos[1])*0.6, 0.04],
                                    self.ee_orientation))
            obs = self.env.operational_position_control(poses)
            self.observations = self.transform_beam_pose(obs)
            current_pos = self.observations["robot_cartesian_pose"][:3]
            poses = np.concatenate(([joint_pos[0], current_pos[1]-(grasped_joint_pos[1]-joint_pos[1]), 0.04],
                                    self.ee_orientation))
            obs = self.env.operational_position_control(poses)
            self.observations = self.transform_beam_pose(obs)

    def assemble_in_f_end(self, grasped_beam, target_beam):
        """
        Assembly for in_f_end joint type
        Assemble beams.
        Args:
            grasped_beam (str): A grasped beam id with a joint id
            target_beamt (str): A target beam id with a joint id
        Returns:
            None
        """

        print("Assemble in_f_end...")
        grasped_beam_id, grasped_joint_id = self.get_beam_joint_id(
            grasped_beam)
        beam_id, joint_id = self.get_beam_joint_id(target_beam)

        joint_pos, joint_ori = mat2pose(
            self.get_beam_joint_pose(beam_id, joint_id))
        grasped_joint_pos, grasped_joint_ori = mat2pose(
            self.get_beam_joint_pose(grasped_beam_id, grasped_joint_id))
        current_pos = self.observations["robot_cartesian_pose"][:3]
        poses = np.concatenate(([current_pos[0]-(grasped_joint_pos[0]-joint_pos[0]), joint_pos[1], 0.038],
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)
        poses = np.concatenate(([current_pos[0]-(grasped_joint_pos[0]-joint_pos[0])*1.25, joint_pos[1], 0.038],
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

    def put_down(self):
        """
        Put down the grasped object
        Args:
            None
        Returns:
            True
        """
        print("Put down the grasped object")

        current_pos = self.observations["robot_cartesian_pose"][:3]

        self.env.gripper_position_control(0.04)  # gripper open

        # Move up
        poses = np.concatenate(([current_pos[0], current_pos[1], 0.3],
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        return True

    def exec_pose_cmd(self, pos, ori=None):
        """
        Execute the cartesian impedance controller
        Args:
            pos: (x, y, z) position
            ori: (x, y, z, w) quaternion
        Returns:
            None
        """
        if ori is None:
            ori = self.ee_orientation
        obs = self.env.operational_force_control(
            np.concatenate((pos, ori)), gripper="close")
        self.observations = self.transform_beam_pose(obs)

    def exec_pose_delta_cmd(self, pos, ori=None):
        """
        Execute the cartesian impedance controller taking delta pose and orientation from the current pose as input.
        Args:
            pos: (x, y, z) delta position
            ori: (x, y, z, w) delta quaternion
        None
        """
        raise NotImplementedError

    def move_neutral_ori(self):
        """
        Rotate the gripper back to the base orientation.
        Args:
            None
        Returns:
            None
        """
        current_pose = self.observations["robot_cartesian_pose"]
        poses = np.concatenate((np.array([current_pose[:3],]),
                                np.array([self.gripper_down,])), axis=1)
        self.observations = self.env.operational_position_control(poses)

    def rotate_gripper(self, current_ori, z_angle):
        """
        Rotate the gripper
        Args:
            current_ori: (x, y, z, w) quaternion of the current orientation
            z_angle: (ax) axis-angle of the z-coordinate
        Returns:
            commanded_quat (array): quarternion of the commanded orientation
        """
        current_pos = self.observations["robot_cartesian_pose"][:3]
        if z_angle == np.pi:
            desired_angles = self.gripper_down_180
        if z_angle == np.pi/2:
            desired_angles = self.gripper_down_90_pos
        pose_delta = np.concatenate((current_pos, desired_angles))
        print("Pose Delta: ", pose_delta)
        obs = self.env.operational_position_control(pose_delta)
        self.observations = self.transform_beam_pose(obs)
        commanded_quat = desired_angles
        return commanded_quat

    def cap_beams(self, grasped_beam, target_beam):
        """
        Capping beams
        Args:
            grasped_beam (str): A grasped beam id with a joint id
            target_beam (str): A target beam id with a joint id
        Returns:
            True
        """

        print("Cap the beams")

        target_beam_id, target_joint_id = self.get_beam_joint_id(target_beam)

        target_beam_pos, target_beam_quat = mat2pose(
            self.get_beam_joint_pose(target_beam_id, target_joint_id)
        )
        grasped_beam_id, grasped_joint_id = self.get_beam_joint_id(
            grasped_beam)
        grasped_joint_pos, grasped_joint_ori = mat2pose(
            self.get_beam_joint_pose(grasped_beam_id, grasped_joint_id))

        initial_pos = self.observations["robot_cartesian_pose"][:3]
        current_pos = self.observations["robot_cartesian_pose"][:3]
        poses = np.concatenate(([current_pos[0], current_pos[1]-(grasped_joint_pos[1]-target_beam_pos[1]), 0.042],
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        current_pos = self.observations["robot_cartesian_pose"][:3]
        poses = np.concatenate(([target_beam_pos[0], current_pos[1]-(grasped_joint_pos[1]-target_beam_pos[1]), 0.05],
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)
        poses = np.concatenate(([target_beam_pos[0], initial_pos[1], 0.05],
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)
        poses = np.concatenate(([target_beam_pos[0], initial_pos[1], 0.04],
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        current_pos = self.observations["robot_cartesian_pose"][:3]
        grasped_joint_pos, grasped_joint_ori = mat2pose(
            self.get_beam_joint_pose(grasped_beam_id, grasped_joint_id))
        target_beam_pos, target_beam_quat = mat2pose(
            self.get_beam_joint_pose(target_beam_id, target_joint_id)
        )
        poses = np.concatenate(([current_pos[0]-(grasped_joint_pos[0]-target_beam_pos[0]), initial_pos[1], 0.047],
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        grasped_joint_pos, grasped_joint_ori = mat2pose(
            self.get_beam_joint_pose(grasped_beam_id, grasped_joint_id))
        target_beam_pos, target_beam_quat = mat2pose(
            self.get_beam_joint_pose(target_beam_id, target_joint_id)
        )
        current_pos = self.observations["robot_cartesian_pose"][:3]
        poses = np.concatenate(([target_beam_pos[0], initial_pos[1], 0.04],
                                self.ee_orientation))
        obs = self.env.operational_position_control(poses)
        self.observations = self.transform_beam_pose(obs)

        return True

    def push(self, beam_id, connected_beam_joint, pre_cap=False):
        """
        Capping a square
        Args:
            beam_id: the first beam whose position needs to be adjusted
            connected_beam_joint: A beam id with a joint id connected to the pushed beam
        """

        print("Push the beam")
        raise NotImplementedError

    def get_beam_origin_pose(self, beam_id):
        """
        Get the origin pose of the given beam id
        Args:
            beam_id (str): A beam id
        Returns:
            origin_pose (array): A 4x4 pose matrix
        """
        beam_info = self.beams[beam_id]
        beam_id_int = int(beam_id[1:])
        marker_pose = self.observations["beams"][beam_id_int-1]
        pos = marker_pose[:3]
        ori = x_quat(marker_pose[3:])
        rot_mat = quat2mat(ori)
        pose_mat = make_pose(pos, rot_mat)
        local_pose_mat = make_pose(beam_info.offset, np.eye(3))
        origin_pose = np.dot(pose_mat, local_pose_mat)

        return origin_pose

    def get_beam_link_pose(self, beam_id, link_id):
        """
        Get the origin pose of the given beam id
        Args:
            beam_id (str): A beam id
            link_id (str): A link id
        Returns:
            link_pose (array): A 4x4 pose matrix
        """

        origin_pose = self.get_beam_origin_pose(beam_id)
        beam_info = self.beams[beam_id]
        local_pose_mat = make_pose(beam_info.link_offsets[link_id], np.eye(3))
        link_pose = np.dot(origin_pose, local_pose_mat)
        return link_pose

    def get_beam_joint_pose(self, beam_id, joint_id=None):
        """
        Calculate the joint poes of the beam
        Args:
            beam_id (str): A beam id with or without a joint id
            joint_id (str): A joint id. Optional if the beam_id contains the joint id
        Returns:
            joint_pose (array): A 4x4 pose matrix
        """

        if joint_id is None:
            beam_id, joint_id = self.get_beam_joint_id(beam_id)

        origin_pose = self.get_beam_origin_pose(beam_id)
        beam_info = self.beams[beam_id]
        local_pose_mat = make_pose(
            beam_info.joint_offsets[joint_id], np.eye(3))
        joint_pose = np.dot(origin_pose, local_pose_mat)

        return joint_pose

    def get_beam_joint_id(self, beam):
        """
        get the beam and joint id
        Args:
            beam (str): a beam id with a joint id
        Returns:
            beam_id (str): a beam id
            joint_id (str): a joint id
        """
        group = re.match("(b\d*)(j\d.*)", beam)
        return group[1], group[2]

    def get_beam_link_id(self, beam):
        """
        get the beam and link id
        Args:
            beam (str): a beam id with a link id
        Returns:
            beam_id (str): a beam id
            link_id (str): a link id
        """
        group = re.match("(b\d*)(l\d.*)", beam)
        return group[1], group[2]

    def get_beam_id(self, beam):
        """
        get the beam id
        Args:
            beam (str): A beam id with a joint or link id
        Returns:
            beam_id (str): a beam id
        """
        group = re.match("(b\d*)(.*)", beam)
        beam_id = group[1]
        return beam_id

    def record_fixed_origin(self):
        """
        Record the fixed beam origin pose and the pose of the marker_id 0.
        """
        self.fixed_beam_origin = self.get_beam_joint_pose(
            self.fixed_beam_id, "j1")
        self.input_origin = np.concatenate((np.array([0.6, 0.2, 0.0]),
                                            self.gripper_down))

    def load_insertion_poses(self, poses):
        """
        Load approach and target poses for insertion.
        Args:
            poses (dict): A dictionary containing the appraoch and target poses for insertion.
                          E.g. {'b1a': pose_for_approach, 'b1t': pose_for_target}
        Returns:
            None
        """
        for key in poses.keys():
            trans = poses[key][:3, 3]
            new_trans = [trans[0], -trans[2], trans[1]]
            euler = quat2axisangle(mat2quat(poses[key][:3, :3]))
            new_quat = quat2mat(axisangle2quat([euler[0], euler[2], euler[1]]))
            new_pose = make_pose(new_trans, new_quat)

            self.beams[key] = np.dot(self.fixed_beam_origin, new_pose)

    def update_beam_approach_pose(self, beam_id, link_id, insert_end):
        """
        Transform the approach poes of the insertion endpoint to the apporach pose of the grasped link
        Args:
            beam_id (str): A beam id
            link_id (str): A link id
            insert_end (str): A string indicating the insertion endpoint
        Returns:
            None
        """
        group = re.match("(b\d*)(j\d*)", insert_end.name)

        approach_pose_wrt_origin = self.beams["{}a".format(beam_id)]
        beam_info = self.beams[beam_id]

        local_pose_mat = make_pose(
            (beam_info.link_offsets[link_id] -
             beam_info.joint_offsets["j1"]), np.eye(3)
        )
        self.beams["{}a".format(beam_id)] = np.dot(
            approach_pose_wrt_origin, local_pose_mat
        )

        target_pose_wrt_origin = self.beams["{}t".format(beam_id)]
        beam_info = self.beams[beam_id]
        self.beams["{}t".format(beam_id)] = np.dot(
            target_pose_wrt_origin, local_pose_mat
        )

    def get_marker_pose(self, marker_id):
        """
        Returns a marker pose in the global coordinate system
        Args:
            marker_id (int): ID of a marker
        """
        raise NotImplementedError

    def transform_beam_pose(self, obs):
        """
        Takes the observation dictionary and transforms the 
        beam poses 90 degrees around the z-axis
        Args:
            obs (dict): Dictionary of observations
        Returns:
            obs (dict): Dictionary of observations with transformed beam poses
        """
        beams = obs["beams"]
        beams_rotated = []

        for beam in beams:
            pos = beam[:3]
            ori = x_quat(beam[3:])
            rot_mat = quat2mat(ori)
            pose_mat = make_pose(pos, rot_mat)
            rot_mat = rotation_matrix(3.14/2, np.array([0, 0, 1]))
            new_pose = np.dot(pose_mat, rot_mat)
            pos, orn = mat2pose(new_pose)
            pose = np.concatenate((pos, w_quat(orn)), axis=0)
            beams_rotated.append(pose)

        obs["beams"] = beams_rotated

        return obs
