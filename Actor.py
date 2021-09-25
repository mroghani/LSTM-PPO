import torch
import torch.nn as nn
from torch import distributions
from torch.nn import functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, hp):
        super().__init__()
        self.hp = hp
        self.lstm = nn.LSTM(state_dim, self.hp.hidden_size, num_layers=self.hp.recurrent_layers)
        self.layer_no_lstm = nn.Linear(state_dim, self.hp.hidden_size) 
        self.layer_hidden = nn.Linear(self.hp.hidden_size, self.hp.hidden_size)
        self.layer_policy_logits = nn.Linear(self.hp.hidden_size, action_dim)
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.log_std_dev = nn.Parameter(hp.init_log_std_dev * torch.ones((action_dim), dtype=torch.float), requires_grad=hp.trainable_std_dev)
        self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0)
        self.hidden_cell = None
        
    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size,self.hp.hidden_size).to(device))
        
    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if self.hp.use_lstm:
            if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
                self.get_init_state(batch_size, device)
            if terminal is not None:
                self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
            _, self.hidden_cell = self.lstm(state, self.hidden_cell)
            hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
            policy_logits_out = self.layer_policy_logits(hidden_out)
        else:
            state = state[-1, :, :]
            hidden_out = F.elu(self.layer_no_lstm(state))
            hidden_out = F.elu(self.layer_hidden(hidden_out))
            policy_logits_out = self.layer_policy_logits(hidden_out)

        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim, self.action_dim) * torch.exp(self.log_std_dev.to(device))
            # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
            policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"), cov_matrix.to("cpu"))
        else:
            policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
        return policy_dist