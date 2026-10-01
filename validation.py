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
    split="val",
    img_wh=(200, 200),
    batch_size=1,
)



#MAJOR THINGS TO TUNE
image_res = 200
batch_size = 1024

camera_angle_x = dataset.camera_angle_x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlp().to(device)


img_num=0
losses=[]


def loss(render, img, batch_num, batch_size=1024, image_res=200):
    indices = range((batch_num-1) *batch_size, min(batch_num*batch_size, image_res**2))
    target_rgb = img.reshape(-1, 3)[indices]
    mse = torch.nn.MSELoss(reduction='mean')
    loss = mse(render, target_rgb)
    losses.append(loss)

def validate():
    for sample in dataset:
        img = sample["image"]

        img_num+=1
        rendered_img = []
        for batch_num in range(1, image_res**2//batch_size +1 ):


            batch_render = torch.zeros([batch_size, 3])
            points, rays_direction, delta = march_rays(sample, camera_angle_x, batch_num, 2,6,64,batch_size)
            flattened = torch.flatten(points, end_dim=1)
            sigma, rgb = model(points, rays_direction)
            print("batch number: ", batch_num, "processed")



            batch_render = render_ray(sigma, rgb, torch.tensor([0,0,0]), 4/63, 64)

            rendered_img.append(batch_render.detach().cpu())
            loss(batch_render, img, batch_num, batch_size)


        #this is for the points left over, in case img res squared doesnt perfectly divide batch size

    
        batch_num+=1
        remainder = image_res**2%batch_size
        batch_render = torch.zeros([remainder, 3])
        points, rays_direction, delta = march_rays(sample, camera_angle_x, batch_num, 2,6,64,remainder)
        flattened = torch.flatten(points, end_dim=1)
        sigma, rgb = model(points, rays_direction)
        
        print("batch number: ", batch_num, "processed")

        batch_render = render_ray(sigma, rgb, torch.tensor([0,0,0]), 4/63, 64)

        rendered_img.append(batch_render.detach().cpu())
        loss(batch_render, img, batch_num, remainder )

        #saving the img

        
        rendered_tensor= torch.cat(rendered_img,dim=0)
        rendered_tensor= rendered_tensor.reshape(image_res, image_res, 3)
        image_np = rendered_tensor.cpu().numpy()
        plt.imshow(image_np)
        plt.axis("off")
        plt.savefig('validation_files/rendered_img'+str(img_num)+".png", bbox_inches='tight', pad_inches=0)
        print("image", img_num, "saved")
        plt.close()  

    print("validation loss:", sum(losses)/len(losses))
