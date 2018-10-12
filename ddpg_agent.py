import torch
import torch.nn.functional as F
import random
import numpy as np
from models import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer
from noise import OUNoise

class DDPGAgent():
    def __init__(self, state_space, action_space, buffer_size, batch_size,learning_rate,update_rate, gamma, tau, device, seed):
        self.action_space = action_space
        self.state_space = state_space
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.step = 0.
        self.update_rate = update_rate
        self.tau = tau
        self.seed = seed
        
        self.actor_local_network = ActorNetwork(state_space, action_space, seed).to(device)
        self.actor_target_network = ActorNetwork(state_space, action_space, seed).to(device)
        self.critic_local_network = CriticNetwork(state_space, action_space, seed).to(device)
        self.critic_target_network = CriticNetwork(state_space, action_space, seed).to(device)
        
        
        self.actor_optimizer = nn.Adam(self.actor_local_network.parameters(), lr=learning_rate)
        self.critic_optimizer = nn.Adam(self.critic_local_network.parameters(), lr=learning_rate)
 
        self.noise = OUNoise(action_space, seed)
        self.memory = ReplayBuffer(action_space = action_space, buffer_size = self.buffer_size, batch_size=self.bacth_size, seed=seed)

        
    def act(self, state, add_noise = True):
        # no epsilon
        state = torch.from_numpy(state).float().to(device)
        self.actor_local_network.eval()
        with torch.no_grad():
            action = self.actor_local_network(state).cpu().data.numpy()
        self.actor_local_network.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1,1)
    
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step = (self.step+1) % update_rate 
        if self.step == 0 and len(self.memory)>self.batch_size:
            self.learn(self.gamma)
            
            
    def learn(self, gamma):
        # interaction between actor & critic network
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        next_actions = self.actor_target_network(next_states)
        q_target_next = self.critic_target_network(next_states,next_actions) 
        q_target = rewards + gamma * q_target_next * (1-dones)
        q_expected = self.critic_local_network(states,actions) 
        critic_loss = F.mse(q_expected, q_target)
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()
        self.soft_update(self.critic_target_network, self.critic_local_network)
        
        actor_preds = self.actor_local_network(states)
        actor_loss = - self.critic_local(states, actor_preds).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.actor_target_network , self.actor_local_network)
       
        
    def soft_update(self, target, local):
        for target_params, local_params in zip(target.parameters(), local.parameters()):
            target_params.data.copy_(self.tau*local_params.data + (1.0-self.tau)*target_params.data)
    
        