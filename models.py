import torch
import torch.functions as F
import numpy as np


class ActorNetwork(torch.nn.Module):
    def __init__(self, state_space, action_space, device, seed):
        super(ActorNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.device = device
        torch.manual_seed(seed)
        
        self.cn1 = torch.nn.Conv2d(self.state_space, 32, kernel_size )
        self.cn2 = torch.nn.Conv2d(32, 63, kernel_size)
        self.fc1 = torch.nn.Linear(,512)
        self.fc2 = torch.nn.Linear(512, action_space)

    def init_weights(self):
        self.cn1.data.weights =
        
    def forward(self, state):
        output = F.relu(self.cn1(state))
        output = F.relu(self.cn2(output))
        output = output.view(output.size(0),-1)
        output = F.relu(self.fc1(output))
        return self.fc2(output)
    
    

class CriticNetwork(torch.nn.Module):
    def __init__(self, action_space, state_space, device, seed):
        super(CriticNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.device = device
        torch.manual_seed(seed)

        self.fc1 = torch.nn.Linear()
        
