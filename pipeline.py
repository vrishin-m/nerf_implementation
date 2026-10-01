#this is the "main program"
#this manages the entire pipeline start to finish.

import torch
import matplotlib.pyplot as plt


from dataset import create_dataloader
from positional_encoding import encode
from ray_generation import march_rays
from renderer import render_ray
from mlp import mlp

dataset, loader = create_dataloader(
    root_dir="data/lego",
    split="train",
    img_wh=(200, 200),
    batch_size=1,
)

print("Number of images:", len(dataset))

image_res = 100
batch_size = 1024

camera_angle_x = dataset.camera_angle_x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlp().to(device)

for sample in dataset:
    rendered_img = torch.zeros([image_res, image_res, 3])

    for batch_num in range(image_res**2//1024 ):
        points, rays_direction, delta = march_rays(sample, camera_angle_x, batch_num, 2,6,64,batch_size)
        flattened = torch.flatten(points, end_dim=1)
        sigma, rgb = model(points, rays_direction)
        print("batch number: ", batch_num, "processed")


        for i in range(batch_size):
            x = (batch_num*batch_size + i)%image_res
            y = (batch_num*batch_size +i)//image_res

            rendered_img[y][x] = render_ray(sigma[i], rgb[i], torch.tensor([0,0,0]), 1/16, 64)


    batch_num+=1
    points, rays_direction, delta = march_rays(sample, camera_angle_x, batch_num, 2,6,64,image_res**2%batch_size)
    flattened = torch.flatten(points, end_dim=1)
    sigma, rgb = model(points, rays_direction)
    print("batch number: ", batch_num, "processed")

    for i in range(image_res**2 % batch_size):
        x = (batch_num*batch_size + i)%image_res
        y = (batch_num*batch_size +i)//image_res

        rendered_img[y][x] = render_ray(sigma[i], rgb[i], torch.tensor([0,0,0]), 1/16, 64)

    image_np = rendered_img.detach().cpu().numpy()
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()
    






