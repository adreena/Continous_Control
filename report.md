## Environment

* Number of agents: 1
* State Space: observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
* Action Space: Each action is a vector with 4 numbers, corresponding to torque applicable to two joints, 
values in action vector must be a number between -1 and 1
* Rewards: +0.1 for each step that the agent's hand is in the goal location
* Goal: maintain its position at the target location for as many time steps as possible

## Algorithm

Recent implementation of “Deep Q Network” (DQN) algorithm has acheived a signficant progress in Reinforcement Learning, resulting in human level performance in playing Atari games.
DQN is capable of solving problems with high-dimensional state space but faces limitations in dealing with high-dimensional continous action space and requires iterative optimazation process at each step.


For this experiment, I used a model-free, off-policy actor-critic algorithm using deep function approximators
that can learn policies in high-dimensional, continuous action spaces and uses some of the deep learning tricks that were introduced along with Deep Q-Networks.
This algorithm is introduced in [Continuous control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) paper and it is built off Deterministic Policy Gradients to produce a policy-gradient actor-critic algorithm called Deep Deterministic Policy Gradients (DDPG).
