"""
---------------------------------------
File: linear_interpolation.py
Credit: This code is based on the robosuite repository (https://github.com/ARISE-Initiative/robosuite)
which is licensed under the MIT license.
Modified by Jack Collins
---------------------------------------
"""


import numpy as np

from core.utils.transforms import *


class LinearInterpolator(object):
    """
    Simple class for implementing a linear interpolator.

    Abstracted to interpolate n-dimensions

    Args:
        ndim (int): Number of dimensions to interpolate

        controller_freq (float): Frequency (Hz) of the controller

        policy_freq (float): Frequency (Hz) of the policy model

        ramp_ratio (float): Percentage of interpolation timesteps across which we will interpolate to a goal position.

            :Note: Num total interpolation steps will be equal to np.floor(ramp_ratio * controller_freq / policy_freq)
                    i.e.: how many controller steps we get per action space update

        ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
            Specified string determines assumed type of input:

                `'euler'`: Euler orientation inputs
                `'quat'`: Quaternion inputs
    """

    def __init__(
        self,
        ndim,
        controller_freq,
        policy_freq,
        ramp_ratio=0.2,
        use_delta_goal=False,
        ori_interpolate=None,
    ):
        self.dim = ndim  # Number of dimensions to interpolate
        self.ori_interpolate = ori_interpolate  # Whether this is interpolating orientation or not
        self.order = 1  # Order of the interpolator (1 = linear)
        self.step = 0  # Current step of the interpolator
        self.total_steps = np.ceil(
            ramp_ratio * controller_freq / policy_freq
        )  # Total num steps per interpolator action
        self.use_delta_goal = use_delta_goal  # Whether to use delta or absolute goals (currently
        # not implemented yet- TODO)
        self.set_states(dim=ndim, ori=ori_interpolate)

    def set_states(self, dim=None, ori=None):
        """
        Updates self.dim and self.ori_interpolate.

        Initializes self.start and self.goal with correct dimensions.

        Args:
            ndim (None or int): Number of dimensions to interpolate

            ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
                Specified string determines assumed type of input:

                    `'euler'`: Euler orientation inputs
                    `'quat'`: Quaternion inputs
        """
        # Update self.dim and self.ori_interpolate
        self.dim = dim if dim is not None else self.dim
        self.ori_interpolate = ori if ori is not None else self.ori_interpolate

        # Set start and goal states
        if self.ori_interpolate is not None:
            if self.ori_interpolate == "euler":
                self.start = np.zeros(3)
            else:  # quaternions
                self.start = np.array((0, 0, 0, 1))
        else:
            self.start = np.zeros(self.dim)
        self.goal = np.array(self.start)

    def set_goal(self, goal, start):
        """
        Takes a requested (absolute) goal and updates internal parameters for next interpolation step

        Args:
            np.array: Requested goal (absolute value). Should be same dimension as self.dim
        """
        # First, check to make sure requested goal shape is the same as self.dim
        if goal.shape[0] != self.dim:
            print("Requested goal: {}".format(goal))
            raise ValueError(
                "LinearInterpolator: Input size wrong for goal; got {}, needs to be {}!".format(goal.shape[0], self.dim)
            )

        # Update start and goal
        self.start = np.array(start)
        self.goal = np.array(goal)

        # Reset interpolation steps
        self.step = 0

    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        NOTE: If this interpolator is for orientation, it is assumed to be receiving either euler angles or quaternions

        Returns:
            np.array: Next position in the interpolated trajectory
        """
        # Grab start position
        x = np.array(self.start)
        # Calculate the desired next step based on remaining interpolation steps
        if self.ori_interpolate is not None:
            # This is an orientation interpolation, so we interpolate linearly around a sphere instead
            goal = np.array(self.goal)
            if self.ori_interpolate == "euler":
                # this is assumed to be euler angles (x,y,z), so we need to first map to quat
                x = mat2quat(euler2mat(x))
                goal = mat2quat(euler2mat(self.goal))

            # Interpolate to the next sequence
            x_current = quat_slerp(x, goal, fraction=(self.step + 1) / self.total_steps)
            if self.ori_interpolate == "euler":
                # Map back to euler
                x_current = mat2euler(quat2mat(x_current))
        else:
            # This is a normal interpolation
            dx = (self.goal - x) / (self.total_steps - self.step)
            x_current = x + dx

        # Increment step if there's still steps remaining based on ramp ratio
        if self.step < self.total_steps - 1:
            self.step += 1
        # Return the new interpolated step
        return x_current