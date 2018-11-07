## Introduction
This project uses Multi-Agent Deep Deterministic Policy Gradient (MADDPG) to train four agents to play [Soccer](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos).


## Environment Description
In this environment we are training four agents to both collaborate and compete in a game of 2v2 soccer.  The Striker's goal is to get the ball into the opponent's goal.  The Goalie's goal is to prevent the ball from entering its own goal.


#### Observation Space
The observation space for each agent consists of 112 variables corresponding to 14 local ray casts, each detecting 7 possible object types, along with the object's distance. Perception is in 180 degree view from front of agent.  Observations over the last three time steps are stacked together for a total of 336 dimensions per agent.  Putting the four agents together results in a final observation vector of 1344 dimensions.


#### Action Space
- Striker: 6 actions corresponding to forward, backward, sideways movement, as well as rotation.
- Goalie: 4 actions corresponding to forward, backward, sideways movement.


#### Reward Structure
- Striker:
  - +1 When ball enters opponent's goal.
  - -0.1 When ball enters own team's goal.
  - -0.001 Existential penalty.
- Goalie:
  - -1 When ball enters team's goal.
  - +0.1 When ball enters opponents goal.
  - +0.001 Existential bonus.


#### Solve Criteria
None specified.


## Installation

#### Step 1: Clone the repo
Clone this repo using `git clone https://github.com/danielnbarbosa/soccer_twos.git`.  Pre-compiled Unity environments for MacOS and Linux are included.


#### Step 2: Install Dependencies
Create an [anaconda](https://www.anaconda.com/download/) environment that contains all the required dependencies to run the project.

##### Mac:
```
conda create --name soccer_twos python=3.6
source activate soccer_twos
conda install -y pytorch -c pytorch
pip install torchsummary tensorboardX unityagents
```

##### Linux:
See separate [instructions](assets/linux_setup.md).


#### Step 3: Download Unity environment
Install the pre-compiled Unity environment.  Select the appropriate file for your operating system:

- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

Download the file into the `unity_envs` directory of this repo and unzip it.

## Train your agent
To train the agent run `./main.py`.  This will start the agent training inside the Unity environment.  Statistics will be output to the command line as well as logged to the 'runs' directory for visualizing via tensorboard.  To start tensorboard run `tensorboard --logdir runs`.

To load a saved model in evaluation mode run `./main.py --eval --load=<path to files>`.  This will load saved weights from checkpoint files.  Evaluation mode disables training and noise, which gives better performance.
