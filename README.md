# Collaboration and Competition

This project is part of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program at Udacity.

## Project Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

![Tennis Environment](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

### State and Action Spaces

- **Observation Space**: The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.
- **Action Space**: Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
- **Reward Structure**: If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

### Solving the Environment

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically:

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Getting Started

### Prerequisites

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Lets---Collab---Compet.git
   cd Lets---Collab---Compet
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Download the Unity environment that matches your operating system:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

   (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.

4. Unzip the downloaded file and place the extracted folder in the root directory of this repository.

## Instructions

### Training the Agent

To train the agent, run:

```
python src/train.py
```

This will train the agent until it achieves an average score of +0.5 over 100 consecutive episodes, and then save the trained model to `checkpoints/tennis_checkpoint.pth`.

### Using the Jupyter Notebook

Alternatively, you can use the Jupyter notebook to train and visualize the agent:

```
jupyter notebook notebooks/Tennis.ipynb
```

Follow the instructions in the notebook to train the agent and visualize its performance.

## Project Structure

- `notebooks/`: Contains Jupyter notebooks for training and visualizing the agent
- `src/`: Contains the source code for the agent and training script
  - `model.py`: Defines the neural network architecture and agent implementation
  - `train.py`: Script for training the agent
- `checkpoints/`: Directory to store trained model weights
- `soccer/`: Optional challenge with the Soccer environment

## Report

For a detailed description of the learning algorithm, neural network architecture, and results, see [Report.md](Report.md).

## Optional Challenge: Soccer Environment

After completing the Tennis environment, you can try the more challenging Soccer environment. See [soccer/README.md](soccer/README.md) for details.
