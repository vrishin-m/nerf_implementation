import torch


def loss(render, img, optimizer, batch_num, batch_size=1024, image_res=200):
    indices = range((batch_num - 1) * batch_size,
                    min(batch_num * batch_size, image_res ** 2))
    target_rgb = img.reshape(-1, 3)[indices]
    mse = torch.nn.MSELoss(reduction='mean')
    mse_loss = mse(render, target_rgb)
    mse_loss.backward()
    optimizer.step()