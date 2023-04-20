from core.env import RAMPEnv
import numpy as np
from core.utils.transforms import *

"""
Minimal working example of the RAMP environment.
"""

if __name__ == '__main__':

    #Set up the environment
    env = RAMPEnv(headless=False, enable_livestream=True)
    obs = env.reset()
    obs = env.step()

    # Inspect the available observations
    for key in obs:
        print(key)
        print(obs[key])

    
    # Example of moving using task-space postion controller
    print("OSC Position Control Example")
    # Create an orientation to use
    gripper_down = euler2mat(np.array([3.14, 0., 3.14]))
    gripper_down = w_quat(mat2quat(gripper_down))

    beam4_pos = obs['beams'][3] #indexed from 0

    pose = np.concatenate((np.array([beam4_pos[0], beam4_pos[1], 0.2]),
                            gripper_down))
    obs = env.operational_position_control(pose)


    # Example of moving through multiple poses using task-space postion controller
    print("OSC Multiple Position Control Example")
    beam5_pos = obs['beams'][4] #indexed from 0

    poses = np.concatenate((np.array([[beam5_pos[0]-0.05, beam5_pos[1], 0.15], 
                                        [beam5_pos[0]-0.05, beam5_pos[1], 0.04], 
                                        [beam5_pos[0]-0.05, beam5_pos[1], 0.15],
                                        [0.4, 0.0, 0.15],]),
                            np.array([gripper_down, 
                                    gripper_down,
                                    gripper_down,
                                    gripper_down,])), axis=1)
    obs = env.operational_position_control(poses)


    # Example of opening and closing the gripper using postion control
    print("Gripper Example")
    env.gripper_position_control(0.04) #open
    env.gripper_position_control(0.005) #close


    # Example using the task-space impedance controller
    print("OSC Impedance Control Example")
    pose = np.concatenate((np.array([beam4_pos[0], beam4_pos[1], 0.2]),
                            gripper_down))
    obs = env.operational_force_control(pose)

    # Close the environment
    env.close()
