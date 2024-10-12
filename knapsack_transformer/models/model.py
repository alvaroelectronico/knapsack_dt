import torch
import torch.nn as nn

class TrajectoryModel(nn.Module):
    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        return torch.zeros_like(actions[-1])
