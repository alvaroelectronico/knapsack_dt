import numpy as np
import torch
import torch.nn as nn
import transformers

from knapsack_transformer.models.model import TrajectoryModel
from knapsack_transformer.models.trajectory_gpt2 import GPT2Model


class KnapsackTransformer(TrajectoryModel):
    def __init__(self, state_dim, act_dim, hidden_size, max_length=None, max_ep_len=4096, action_tanh=True, **kwargs):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Prediction heads
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, self.act_dim),
            nn.Sigmoid()  # For binary action (select/reject)
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Combine embeddings with time embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack inputs
        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Transformer forward pass
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs, attention_mask=attention_mask)
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Predictions
        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, returns_to_go, timesteps, **kwargs):
        # Reshape inputs for batch processing
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            attention_mask = torch.cat(
                [torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])]).to(
                states.device).reshape(1, -1)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(states, actions, returns_to_go, timesteps, attention_mask=attention_mask)
        return action_preds[0, -1]
