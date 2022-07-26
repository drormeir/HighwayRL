import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, optim, nn


class HighwayActorModelDQN(nn.Module):
    def __init__(self, state_size, action_size, model_params, seed: int = 0, pytorch_device=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            pytorch_device (str): pytorch cuda device name
        """
        super().__init__()
        self.model_params = model_params
        ch_conv1, ch_conv2 = model_params['ch_conv1'], model_params['ch_conv2']
        fc0_out, fc1_out, fc2_out = model_params['fc0_out'], model_params['fc1_out'], model_params['fc2_out']
        self.seed = seed
        self.torch_seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.conv = nn.Sequential(
            nn.Conv2d(self.state_size[0], ch_conv1, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(ch_conv1, ch_conv2, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        feature_dim = self.conv(autograd.Variable(torch.zeros(1, *self.state_size))).view(1, -1).size(1)
        self.base_stream = nn.Sequential(
            nn.Linear(feature_dim, fc0_out),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(fc0_out, fc1_out),
            nn.ReLU(),
            nn.Linear(fc1_out, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc0_out, fc2_out),
            nn.ReLU(),
            nn.Linear(fc2_out, self.action_size)
        )
        self.apply(HighwayActorModelDQN.init_weights)

        self.pytorch_device = pytorch_device
        if isinstance(pytorch_device, str):
            pytorch_device = pytorch_device.lower()
            if any(pytorch_device.startswith(device_name) for device_name in ['gpu', 'cuda']):
                if torch.cuda.is_available():
                    self.to(torch.device('cuda:0'))

    def clone(self):
        ret = HighwayActorModelDQN(state_size=self.state_size, action_size=self.action_size,
                                   model_params=self.model_params,
                                   seed=self.seed, pytorch_device=self.pytorch_device)
        ret.soft_update_from_local(self, 1.0)
        return ret

    @property
    def my_device(self):
        return next(self.parameters()).device

    def act(self, state):
        self.eval()
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.my_device)
        with torch.no_grad():
            action_values = self(state).cpu().data.numpy()
        return int(np.argmax(action_values))

    def q_targets(self, rewards, gamma, dones, next_states):
        self.eval()
        rewards = torch.from_numpy(rewards).float().to(self.my_device)
        dones = torch.from_numpy(dones).int().to(self.my_device)
        next_states = torch.from_numpy(next_states).float().to(self.my_device)
        # Get max predicted Q values (for next states) from target model
        q_targets_next = self(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + gamma * (1 - dones) * q_targets_next
        return q_targets

    def q_expected(self, states, actions):
        self.train()
        # Get expected Q values from local model
        states = torch.from_numpy(states).float().to(self.my_device)
        actions = torch.from_numpy(actions).long().to(self.my_device)
        q_expected = self(states).gather(1, actions)
        return q_expected

    def soft_update_from_local(self, local_q_network, local_tau_weight):
        for target_param, local_param in zip(self.parameters(), local_q_network.parameters()):
            target_param.data.copy_(local_tau_weight * local_param.data + (1.0 - local_tau_weight) * target_param.data)

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        features = self.base_stream(features)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean())
        return q_vals

    def save(self, filename):
        shutil.rmtree(filename, ignore_errors=True)  # avoid file not found error
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class OptimizerDQN:
    def __init__(self, local_q_network, lr):
        self.optimizer = optim.Adam(local_q_network.parameters(), lr=lr)

    def step(self, q_expected, q_targets):
        if isinstance(q_targets, list):
            q_targets = torch.min(torch.stack(q_targets), dim=0)[0]  # [0] to get the values
        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        shutil.rmtree(filename, ignore_errors=True)  # avoid file not found error
        torch.save(self.optimizer.state_dict(), filename)

    def load(self, filename):
        self.optimizer.load_state_dict(torch.load(filename))

