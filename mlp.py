import torch
from torch import nn
from positional_encoding import encode




hidden_size = 128


class mlp(nn.Module):

    def __init__(self):
        super().__init__()
        self.L_pos = 10
        self.L_dir = 4
        self.sigma_network = nn.Sequential(
            nn.Linear(6 * self.L_pos, hidden_size),
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
            nn.Linear(hidden_size + 6 *self.L_dir, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),
            nn.Sigmoid()
        )

        self.project_sigma = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

    def forward(self, positions, directions):
        position, direction = encode(positions, directions, self.L_pos, self.L_dir)
        sigma_embedding = self.sigma_network(position)

        rgb_input = torch.cat(
            (sigma_embedding, direction),
            dim=-1
        )

        rgb = self.rgb_network(rgb_input)

        sigma = self.project_sigma(sigma_embedding)

        return sigma, rgb

    