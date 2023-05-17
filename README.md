# RAMP - Sim: A Benchmark for Evaluating Robotic Assembly Manipulation and Planning
A repository containing the simulation component of the RAMP benchmark

![Teaser figure](./media/simulation.png)

## Dependencies

There are two dependencies for ramp-sim:
- Nvidia Isaac 2022.2.1
- The Planner found [here](https://github.com/applied-ai-lab/planner)

## Installation

There are two options for installing and running the simulation, (1) locally, or (2) in a Docker container.

### Locally 

1. Clone this repository
```
git clone --recursive --branch main git@github.com:applied-ai-lab/ramp-sim.git
cd ramp-sim
git submodule update --init --recursive
```
2. Install Nvidia Isaac locally following the guide [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html)
3. Follow the instructions to install the planner from [here](https://github.com/applied-ai-lab/planner)

### Docker 

1. Clone this repository
```
git clone --recursive --branch main git@github.com:applied-ai-lab/ramp-sim.git
cd ramp-sim
git submodule update --init --recursive
```
2. Follow the guide to install the containerised Nvidia Isaac from [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_container.html)
3. Build the docker container from the dockerfile 
```
docker build -t isaac_ramp .
```

## To Run

For convenience there is a scripts/simple_assembly.py that demonstrates how to use the simulation environment. For the scripts/main.py there is a single flag that is used to decide which assembly to attempt. The flag ```--assembly``` takes as an argument the XML file for desired assembly, currently either assembly_easy_1.xml, assembly_easy_2.xml or assembly_easy_3.xml.

### Locally

```
/home/$USER/pkg/isaac_sim-2022.2.1/python.sh scripts/main.py --assembly assembly_easy_2.xml
```

### Docker

```
./docker/run_docker.sh
/isaac-sim/python.sh scripts/main.py --assembly assembly_easy_2.xml
```




## Simulation environment improvements to be made

1. The joint limits have been increased for task-based motion planning, otherwise, the Nvidia Isaac motion planner often fails to generate feasible trajectories. 
2. Transforms in /core/utils/transforms.py to be w,x,y,z
3. Fiducial markers to be added to the joints
4. Add noise to observations
5. Incorporate April Tag detection
6. Instead of relying on a pre-built USD, create a script for procedural generation
   1. Requires finger tips to be added to the Panda
   2. Requires ee xform and rigid joint to be added to Panda
   3. Seperately load each of the meshes, add joints, SDF collisions, etc.


## Cite

An arXiv paper is available [here](https://arxiv.org/abs/2305.09644), and can be cited with the following bibtex entry:

```
@misc{collins2023ramp,
      title={RAMP: A Benchmark for Evaluating Robotic Assembly Manipulation and Planning}, 
      author={Jack Collins and Mark Robson and Jun Yamada and Mohan Sridharan and Karol Janik and Ingmar Posner},
      year={2023},
      eprint={2305.09644},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

