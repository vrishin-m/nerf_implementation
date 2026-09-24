#this is the mlp thats going to give colors, sigma from position, view angle

import torch
from torch import nn
from torch.utils.data import DataLoader
from positional_encoding import encode
from positional_encoding import L_dir,  L_pos

hidden_size =128
position, direction = encode(input)


class mlp(nn.module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sigma_network = nn.Sequential(
            nn.Linear(6*L_pos, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.rgb_network = nn.Sequential(
            nn.Linear(hidden_size+4*L_dir, hidden_size/2),
            nn.ReLU(),
            nn.Linear(hidden_size/2, 3)

        )

        self.project_sigma = nn.Sequential(
            nn.Linear(hidden_size,1),
            nn.ReLU()
        )

    def forward(self, position, direction):
        position = self.flatten(position)
        sigma_embedding = self.sigma_network(position)
        rgb = self.rgb_network(torch.cat((sigma_embedding, direction)))
        sigma = self.project_sigma(sigma_embedding)
        return sigma,rgb