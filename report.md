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

### Agent

DDPG agent consists of 4 networks actor_network, actor_target_network, critic_network and critic_target_network. It starts by taking actions in epsilon-greedy manner and adding tuple of <state, action, reward, next_action, done> to its replay buffer. At every 100 steps (i.e. update_rate) it does a learning process by updating its local actor & critic network which includes backpropagation steps through each network to calculate gradients and finally applying a soft update to the target networks.

Actor nework has a hidden layer of 128 nodes and outputs actions for each state. Critic network consists of 3 hidden layers with 128, 64 & 32 neurons which outputs a single value. both architectures use learning rate of 0.001 and Adam optimizer.

Other hyperparameters are:
  - buffer_size: 100,000 tuples, 
  - batch_size: 512
  - learning_rate_actor: 0.001
  - learning_rate_critic: 0.001
  - gamma (discount factor): 0.995
  - tau (soft update interpolation rate): 0.001
  
Agent is trained over 10,000 episodes and the average of 100 last episodes are plotted below:

