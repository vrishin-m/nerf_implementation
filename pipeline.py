#this is the "main program"
#this manages the entire pipeline start to finish.

import torch
import matplotlib.pyplot as plt
import torch.optim as optim

from dataset import create_dataloader
from positional_encoding import encode
from ray_generation import march_rays
from renderer import render_ray
from mlp import mlp
from loss import loss


dataset, loader = create_dataloader(
    root_dir="data/lego",
    split="train",
    img_wh=(200, 200),
    batch_size=1,
)

print("Number of images:", len(dataset))

image_res = 200
batch_size = 1024

camera_angle_x = dataset.camera_angle_x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlp().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

img_num=0


for sample in dataset:
    img = sample["image"]
    print(img.shape)
    img_num+=1
    rendered_img = []
    for batch_num in range(1, image_res**2//batch_size +1 ):
        optimizer.zero_grad()

        batch_render = torch.zeros([batch_size, 3])
        points, rays_direction, delta = march_rays(sample, camera_angle_x, batch_num, 2,6,64,batch_size)
        flattened = torch.flatten(points, end_dim=1)
        sigma, rgb = model(points, rays_direction)
        print(rgb.min(), rgb.max(), rgb.mean())
        print(sigma.min(), sigma.max(), sigma.mean())
        print("batch number: ", batch_num, "processed")


        for i in range(batch_size):
            batch_render[i] = render_ray(sigma[i], rgb[i], torch.tensor([0,0,0]), 1/16, 64)

        rendered_img.append(batch_render.detach().cpu())
        loss(batch_render, img, optimizer, batch_num, batch_size, image_res)


    #this is for the points left over, in case img res squared doesnt perfectly divide batch size

    optimizer.zero_grad()
    batch_num+=1
    remainder = image_res**2%batch_size
    batch_render = torch.zeros([remainder, 3])
    points, rays_direction, delta = march_rays(sample, camera_angle_x, batch_num, 2,6,64,remainder)
    flattened = torch.flatten(points, end_dim=1)
    sigma, rgb = model(points, rays_direction)
    print("batch number: ", batch_num, "processed")



    for i in range(remainder):
        batch_render[i] = render_ray(sigma[i], rgb[i], torch.tensor([0,0,0]), 1/16, 64)


    rendered_img.append(batch_render.detach().cpu())
    loss(batch_render, img, optimizer, batch_num, batch_size, image_res)

    #saving the img

    
    rendered_tensor= torch.cat(rendered_img,dim=0)
    rendered_tensor= rendered_tensor.reshape(image_res, image_res, 3)
    image_np = rendered_tensor.cpu().numpy()
    plt.imshow(image_np)
    plt.axis("off")
    plt.savefig('images/rendered_img'+str(img_num)+".png", bbox_inches='tight', pad_inches=0)
    plt.close()  

 
    





    






