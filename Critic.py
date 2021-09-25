import torch
import torch.nn as nn
from torch import distributions
from torch.nn import functional as F
   
class Critic(nn.Module):
    def __init__(self, state_dim, hp):
        super().__init__()
        self.hp = hp
        self.layer_lstm = nn.LSTM(state_dim, self.hp.hidden_size, num_layers=self.hp.recurrent_layers)
        self.layer_no_lstm = nn.Linear(state_dim, self.hp.hidden_size) 
        self.layer_hidden = nn.Linear(self.hp.hidden_size, self.hp.hidden_size)
        self.layer_value = nn.Linear(self.hp.hidden_size, 1)
        self.hidden_cell = None
        
    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.hidden_size).to(device))
    
    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if self.hp.use_lstm:
            if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
                self.get_init_state(batch_size, device)
            if terminal is not None:
                self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
            _, self.hidden_cell = self.layer_lstm(state, self.hidden_cell)
            hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
            value_out = self.layer_value(hidden_out)
        else:
            state = state[-1, :, :]
            hidden_out = F.elu(self.layer_no_lstm(state))
            hidden_out = F.elu(self.layer_hidden(hidden_out))
            value_out = self.layer_value(hidden_out)
        return value_out