import torch
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(torch.nn.Module):
    def __init__(self, state_space, action_space, device, seed):
        super(ActorNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.device = device
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = torch.nn.Linear(self.state_space, 128 )
        self.fc2 = torch.nn.Linear(128, action_space)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        output = F.relu(self.fc1(state))
        return F.tanh(self.fc2(output))

class CriticNetwork(torch.nn.Module):
    def __init__(self, state_space, action_space, device, seed):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_space = state_space
        self.action_space = action_space
        self.device = device

        self.fc1 = torch.nn.Linear(state_space, 128)
        self.fc2 = torch.nn.Linear(128+action_space, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 1)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        output = F.relu(self.fc1(state))
        output = torch.cat((output, action), dim=1)
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        return self.fc4(output)
        
        
