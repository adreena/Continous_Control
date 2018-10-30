
from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import DDPGAgent
import torch
from collections import deque, namedtuple
from tensorboardX import SummaryWriter 
import argparse

def run(env, device, episodes, experiment_name, update_rate, action_size, state_size, brain_name,
        epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, max_score = 30., num_agents=1):

    epsilon = epsilon_start
    
    agent = DDPGAgent(state_space=state_size, action_space=action_size, buffer_size=int(1e5), 
                      batch_size=512,learning_rate_actor=0.001, learning_rate_critic=0.001, update_rate=update_rate, 
                      gamma=0.995, tau=0.001, device=device, seed=5, num_agents = num_agents)
    score_window = deque(maxlen=100)
    all_scores = []
    tb_writer = SummaryWriter('{}/{}'.format('logs',experiment_name))
    
    for episode in range(episodes):
        agent.reset()
        scores = np.zeros(num_agents) 
        dones = np.zeros((num_agents), dtype=bool)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations 
        while not np.any(dones):
            actions = agent.act(states, epsilon)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)
            
            states = next_states
            
        
        episode_score = np.mean(scores)
        score_window.append(episode_score)
        all_scores.append(episode_score)
        
        #decay after each episode
        if episode % 10 == 0:
            epsilon = max(epsilon_min, epsilon*epsilon_decay)
            
        print('\rEpisode: {}\tAverage Score: {}'.format(episode, np.mean(score_window)), end="")
        if episode % 100 == 0:
            tb_writer.add_scalar('Episode_Accum_score', np.mean(score_window), episode)

            print('\rEpisode: {}\tAverage Score: {}'.format(episode, np.mean(score_window)))
        if np.mean(score_window) >= max_score:
            torch.save(agent.actor_local_network.state_dict(), 'actor_checkpoint_{}.pth'.format(experiment_name))
            torch.save(agent.critic_local_network.state_dict(), 'critic_checkpoint_{}.pth'.format(experiment_name))
            break


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='type experiment name')
    parser.add_argument('--rate', help='update_rate')
    parser.add_argument('--cuda', help='cuda index')
    parser.add_argument('--episodes', help='number of episodes')
    args = parser.parse_args()
    print(args)
    
    
    env = UnityEnvironment(file_name='Reacher_Linux_NoVis_{}/Reacher.x86_64'.format(args.cuda))
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    run(env=env, device=device, episodes=int(args.episodes), experiment_name=args.experiment, update_rate=int(args.rate), \
        brain_name=brain_name, num_agents=num_agents, action_size=action_size,state_size=state_size )
    print('finished')