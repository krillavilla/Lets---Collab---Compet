# Soccer Environment

This directory contains the optional Soccer challenge for the Collaboration and Competition project.

## Unity Environment

To use this environment, you need to download the Unity Soccer environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

For AWS (without a virtual screen): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux_NoVis.zip)

## Environment Details

In this environment, the goal is to train a team of agents to play soccer.

The environment contains two separate agents:
- Striker agents: Responsible for scoring goals
- Goalie agents: Responsible for defending goals

Each agent receives its own, local observation of the environment. The observation space consists of 112 variables corresponding to local information about the field and players.

The action space is discrete with 3 possible actions:
1. Move forward/backward/stay still
2. Rotate clockwise/counterclockwise/stay still
3. Jump/don't jump

## Getting Started

Follow the instructions in the `notebooks/Soccer.ipynb` notebook to get started with training your own agents for the Soccer environment.