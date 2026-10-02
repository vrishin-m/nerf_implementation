#this is the "main program"
#this manages the entire pipeline start to finish.

import os
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

from dataset import create_dataloader
from positional_encoding import encode
from ray_generation import march_rays
from renderer import render_ray
from mlp import mlp
from loss import loss
from validation import validate

dataset, loader = create_dataloader(
    root_dir="data/lego",
    split="train",
    img_wh=(200, 200),
    batch_size=1,
)

print("Number of images:", len(dataset))

#MAJOR THINGS TO TUNE
image_res = 200
batch_size = 1024
learning_rate= 5e-4
camera_angle_x = dataset.camera_angle_x
near = 2.0
far  = 8.0
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlp(near=near, far=far).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs("images", exist_ok=True)

img_num = 0
start_epoch = 0
start_img_idx = 0

latest_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
if os.path.exists(latest_ckpt_path):
    print(f"Loading checkpoint from {latest_ckpt_path}...")
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    saved_epoch = checkpoint.get("epoch", 0)
    img_num = checkpoint.get("img_num", 0)
    saved_img_in_epoch = checkpoint.get("img_in_epoch", img_num % len(dataset))
    learning_rate = checkpoint.get("learning_rate", learning_rate)

    if saved_img_in_epoch == 0 and img_num > 0:
        saved_img_in_epoch = len(dataset)

    if saved_img_in_epoch >= len(dataset):
        start_epoch = saved_epoch + 1
        start_img_idx = 0
    else:
        start_epoch = saved_epoch
        start_img_idx = saved_img_in_epoch

    print(f"Resumed from epoch {start_epoch} (next image index {start_img_idx + 1}), total images processed: {img_num}")

for epoch in range(start_epoch, epochs):
    for idx, sample in enumerate(dataset):
        if epoch == start_epoch and idx < start_img_idx:
            continue
        learning_rate -= 0.03e-4
        img = sample["image"]
        print(img.shape)
        img_num += 1
        img_in_epoch = idx + 1
        rendered_img = []
        for batch_num in range(1, image_res**2//batch_size +1 ):
            optimizer.zero_grad()

            batch_render = torch.zeros([batch_size, 3])
            points, rays_direction, delta = march_rays(sample, camera_angle_x, batch_num, near, far, 128, batch_size)
            sigma, rgb = model(points, rays_direction)
            print("rgb miin,max, mean", rgb.min(), rgb.max(), rgb.mean())
            print("sigma min, max, mean", sigma.min(), sigma.max(), sigma.mean())
            print("batch number: ", batch_num, "processed")


            
            batch_render = render_ray(sigma, rgb, torch.tensor([1.0,1.0,1.0]), 6/127, 128)

            rendered_img.append(batch_render.detach().cpu())
            loss(batch_render, img, optimizer, batch_num, batch_size, image_res)


        #this is for the points left over, in case img res squared doesnt perfectly divide batch size

        optimizer.zero_grad()
        batch_num+=1
        remainder = image_res**2%batch_size
        batch_render = torch.zeros([remainder, 3])
        points, rays_direction, delta = march_rays(sample, camera_angle_x, batch_num, near, far, 128, remainder)
        sigma, rgb = model(points, rays_direction)

        print("batch number: ", batch_num, "processed")

        batch_render = render_ray(sigma, rgb, torch.tensor([1.0,1.0,1.0]), 6/127, 128)

        rendered_img.append(batch_render.detach().cpu())
        loss(batch_render, img, optimizer, batch_num, batch_size, image_res)

        #saving the img

        
        rendered_tensor= torch.cat(rendered_img,dim=0)
        rendered_tensor= rendered_tensor.reshape(image_res, image_res, 3)
        image_np = rendered_tensor.cpu().numpy()
        plt.imshow(image_np)
        plt.axis("off")
        image_save_path = f"images/rendered_epoch_{epoch}_img_{img_in_epoch}.png"
        plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)
        print(f"Image saved at {image_save_path} (total: {img_num})")
        plt.close()  

        # Checkpoint every 10 images
        if img_in_epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_img_{img_in_epoch}.pth")
            checkpoint_data = {
                "epoch": epoch,
                "img_num": img_num,
                "img_in_epoch": img_in_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "learning_rate": learning_rate,
            }
            torch.save(checkpoint_data, checkpoint_path)
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, "checkpoint_latest.pth"))
            print(f"Checkpoint saved at {checkpoint_path}")  

    print("\n" * 3, "____________________________________________________")
    print(f"EPOCH {epoch} done. Running evaluation...")
    validate(model=model, epoch=epoch, device=device, near=near, far=far)
    model.train()
