import torch
from torch import nn
import torch.nn.functional as F
from positional_encoding import encode




hidden_size = 256
pos_enc_dim = 60   
dir_enc_dim = 24   



class mlp(nn.Module):

    def __init__(self, near=2.0, far=8.0):
        super().__init__()
        self.L_pos = 10
        self.L_dir = 4

        self.near = near
        self.far  = far

       
        self.sigma_net_1 = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    
        self.sigma_net_2 = nn.Sequential(
            nn.Linear(hidden_size + pos_enc_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

      
        self.rgb_network = nn.Sequential(
            nn.Linear(hidden_size + dir_enc_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),
        )


        self.project_sigma = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.project_sigma.bias, 0.1)

    def forward(self, positions, directions):
        scene_center = (self.near + self.far) / 2.0
        scene_scale  = (self.far  - self.near) / 2.0
        positions_norm = (positions - scene_center) / scene_scale

        position, direction = encode(positions_norm, directions, self.L_pos, self.L_dir)

        h = self.sigma_net_1(position)
        h = torch.cat([h, position], dim=-1)  
        sigma_embedding = self.sigma_net_2(h)

     
        rgb_input = torch.cat((sigma_embedding, direction), dim=-1)
        rgb = torch.sigmoid(self.rgb_network(rgb_input))

        sigma = F.softplus(self.project_sigma(sigma_embedding))

        return sigma, rgb