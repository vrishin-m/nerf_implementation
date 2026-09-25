
import torch


L_pos = 10
L_dir = 4


def encode(positions, directions):
    gammapos = []
    gammadir = []

    for i in range(L_pos):
        frequency = 2 ** i * torch.pi

        for coordinate in range(3):
            value = positions[:, coordinate]

            gammapos.append(torch.sin(frequency * value))
            gammapos.append(torch.cos(frequency * value))


    for i in range(L_dir):
        frequency = 2 ** i * torch.pi

        for coordinate in range(2):
            value = directions[:, coordinate]

            gammadir.append(torch.sin(frequency * value))
            gammadir.append(torch.cos(frequency * value))


    gammapos = torch.stack(gammapos, dim=1)
    gammadir = torch.stack(gammadir, dim=1)

    return gammapos, gammadir








