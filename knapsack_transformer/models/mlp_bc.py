import torch
import torch.nn as nn
from knapsack_transformer.models.model import TrajectoryModel

class MLPBCModel(TrajectoryModel):
    """
    MLP that predicts the next action for the knapsack problem based on past states.
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__(state_dim, act_dim)

        layers = [nn.Linear(max_length * self.state_dim, hidden_size)]
        for _ in range(n_layer - 1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Sigmoid(),  # Sigmoid for binary action (select/reject)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):
        states = states[:, -self.max_length:].reshape(states.shape[0], -1)  # Flatten state sequence
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)
        return None, actions, None

    def get_action(self, states, actions, rewards, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length - states.shape[1], self.state_dim), dtype=torch.float32,
                             device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0, -1]
