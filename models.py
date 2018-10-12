import torch
import torch.nn.functional as F
import numpy as np


class ActorNetwork(torch.nn.Module):
    def __init__(self, state_space, action_space, device, seed):
        super(ActorNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.device = device
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = torch.nn.Linear(self.state_space, 512 )
        self.fc2 = torch.nn.Linear(512, action_space)

    def forward(self, state):
        output = F.relu(self.fc1(state))
        return F.tanh(self.fc2(output))


class CriticNetwork(torch.nn.Module):
    def __init__(self, action_space, state_space, device, seed):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_space = state_space
        self.action_space = action_space
        self.device = device

        self.fc1 = torch.nn.Linear(state_space, 100)
        self.fc2 = torch.nn.Linear(100+action_space, 512)
        self.fc3 = torch.nn.Linear(512, 1)
        
    def forward(self, state, action):
        output = F.leaky_relu(self.fc1(state))
        output = torch.cat((output, action), dim=1)
        output = F.leaky_relu(self.fc2(output))
        return self.fc3(output)

        
