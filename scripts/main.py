import os
import sys
import numpy as np
import argparse

from core.env import RAMPEnv

from core.agent import Agent
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the RAMP simulator with the planner for a given assembly.')
    parser.add_argument("-a", "--assembly", type=str,
                        required=True, help='The assembly XML file to use.')

    args = parser.parse_args()
    assembly = args.assembly

    if assembly == "assembly_easy_1.xml":
        from sparc_planning.src.example_domains.example_easy_1 import generate_domain_setup
    elif assembly == "assembly_easy_2.xml":
        from sparc_planning.src.example_domains.example_easy_2 import generate_domain_setup
    elif assembly == "assembly_easy_3.xml":
        from sparc_planning.src.example_domains.example_easy_3 import generate_domain_setup

    print('Plan...')

    agent = Agent(beam_xml=os.path.join(os.environ['PLANNER_PATH'], 'example_beamset_latest.xml'),
                  assembly_xml=os.path.join(
                      os.environ['PLANNER_PATH'], assembly),
                  domain_generator=generate_domain_setup)

    coarse_actions, fine_actions, fine_actions_per_coarse = agent.plan()

    print('Setup virtual environment...')

    my_env = RAMPEnv(headless=True, enable_livestream=True)
    obs = my_env.reset()
    obs = my_env.step()
    agent.interface.env = my_env
    agent.interface.observations = obs

    print('Execute...')

    agent.execute(coarse_actions, fine_actions, fine_actions_per_coarse)

    my_env.close()
