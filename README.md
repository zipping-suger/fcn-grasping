# fcn-grasping
---
FCN based one-shot grasping
## Installation
Tested on Ubuntu 20.04.4 LTS with the following dependencies:
* Python 3.6
* CoppeliaSim simulation environment (https://www.coppeliarobotics.com/downloads)
* PyTorch version 1.0+ 
For GeForce RTX 3080,
```commandline
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Other python package dependencies could be installed with
```commandline
pip3 install -r requirements.txt 
```

## Run

1. **Launch CoppeliaSim:** navigate to the CoppeliaSim directory and run `./coppeliaSim.sh` . From the main menu, select File > Open scene..., and open the file `fcn-grasping/simulation/simulation_barrett.ttt` from this repository. 
2. Run `train_multimodal.py` or `train_reconfigurable.py`


## Script Description
1. **NN_models.py**  defines the structures of FCNs using one feature extraction trunk and multiple output branches. (Input color and depth height map) 
   1. `HybridNet` outputs grasping qualities and grasping configurations.
   2. `MultiQNet` outputs grasping qualities for different grasping modes.
      1. `TeacherNet` uses deep feature extraction backbone (densenet121)
      2. `StudentNet` uses shallow feature extraction backbone (resnet18)

2. **trainer.py** defines the training procedures of different network models.
3. **robot.py** sets up the UR5 robot and camera in realworld and simulation. It also provides API for grasping primitives.
4. **train_multimodal.py** is the main function to train the `MultiQNet` for grasping, while 'train_reconfigurable.py' trains the `HybridNet`.