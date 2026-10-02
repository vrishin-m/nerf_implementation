import os
import math
import torch
import matplotlib.pyplot as plt

from dataset import create_dataloader
from ray_generation import march_rays
from renderer import render_ray
from mlp import mlp

val_dataset, val_loader = create_dataloader(
    root_dir="data/lego",
    split="val",
    img_wh=(200, 200),
    batch_size=1,
)


def compute_eval_loss(render, img, batch_num, batch_size=1024, image_res=200):
    indices = range((batch_num - 1) * batch_size, min(batch_num * batch_size, image_res ** 2))
    target_rgb = img.reshape(-1, 3)[indices]
    mse = torch.nn.MSELoss(reduction='mean')
    return mse(render, target_rgb).item()


def validate(
    model=None,
    epoch=0,
    dataset=None,
    device=None,
    near=2.0,
    far=8.0,
    samples=128,
    batch_size=1024,
    image_res=200,
    save_dir="validation_files",
    max_samples=None,
):
   
    if device is None:
        if model is not None:
            device = next(model.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = mlp(near=near, far=far).to(device)
        latest_ckpt = "checkpoints/checkpoint_latest.pth"
        if os.path.exists(latest_ckpt):
            print("loading weights")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get("epoch", epoch)
        else:
            print("no checkpoint found")

    was_training = model.training
    model.eval()

    eval_dataset = dataset if dataset is not None else val_dataset
    camera_angle_x = getattr(eval_dataset, "camera_angle_x", 0.6911112070083618)
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    delta = (far - near) / (samples - 1)

    os.makedirs(save_dir, exist_ok=True)

    val_losses = []
    val_psnrs = []

    print(f"\n evaluation for epoch {epoch} ({len(eval_dataset)} images)")

    with torch.no_grad():
        for val_idx, sample in enumerate(eval_dataset):
            if max_samples is not None and val_idx >= max_samples:
                break

            val_img_num = val_idx + 1
            img = sample["image"]

            rendered_img = []
            num_full_batches = (image_res ** 2) // batch_size
            remainder = (image_res ** 2) % batch_size

            for batch_num in range(1, num_full_batches + 1):
                points, rays_direction, _ = march_rays(
                    sample, camera_angle_x, batch_num, near, far, samples, batch_size
                )
                points = points.to(device)
                rays_direction = rays_direction.to(device)
                sigma, rgb = model(points, rays_direction)

                batch_render = render_ray(sigma, rgb, bg_color, delta, samples)
                rendered_img.append(batch_render.detach().cpu())

            if remainder > 0:
                batch_num = num_full_batches + 1
                points, rays_direction, _ = march_rays(
                    sample, camera_angle_x, batch_num, near, far, samples, remainder
                )
                points = points.to(device)
                rays_direction = rays_direction.to(device)
                sigma, rgb = model(points, rays_direction)

                batch_render = render_ray(sigma, rgb, bg_color, delta, samples)
                rendered_img.append(batch_render.detach().cpu())

            rendered_tensor = torch.cat(rendered_img, dim=0)
            target_tensor = img.reshape(-1, 3).cpu()

            img_mse = torch.mean((rendered_tensor - target_tensor) ** 2).item()
            img_psnr = -10.0 * math.log10(img_mse) if img_mse > 0 else 99.0

            val_losses.append(img_mse)
            val_psnrs.append(img_psnr)

            # Save rendered image
            rendered_image = torch.clamp(rendered_tensor, 0.0, 1.0).reshape(image_res, image_res, 3).numpy()
            plt.imshow(rendered_image)
            plt.axis("off")
            image_save_path = os.path.join(save_dir, f"val_epoch_{epoch}_img_{val_img_num}.png")
            plt.savefig(image_save_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            print(f"Validation Image [{val_img_num}/{len(eval_dataset)}] saved: {image_save_path} | MSE: {img_mse:.5f} | PSNR: {img_psnr:.2f} dB")

    avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
    avg_psnr = sum(val_psnrs) / len(val_psnrs) if val_psnrs else 0.0

    print("=" * 60)
    print(f"EPOCH {epoch} EVALUATION SUMMARY:")
    print(f"Average Val Loss (MSE): {avg_loss:.6f} | Average Val PSNR: {avg_psnr:.2f} dB")
    print("=" * 60 + "\n")

    if was_training:
        model.train()

    return {"loss": avg_loss, "psnr": avg_psnr}



loss = compute_eval_loss

if __name__ == "__main__":
    validate()
