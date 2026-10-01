import torch


def render_ray(sigma, rgb, color_bg = torch.zeros([3]), delta=4/63, samples=64):
    sigma = sigma.squeeze(-1)
    print("one ray", sigma[0])
    print(sigma.shape, rgb.shape)
    t_cum=torch.ones(sigma.shape[0])
    color = torch.zeros([sigma.shape[0],3])
    for i in range(samples):
        t= torch.exp(-sigma[...,i]*delta)
        alpha = torch.ones(sigma.shape[0])-t
        w = t_cum * alpha
        t_cum = t_cum* t
        color += rgb[:,i,:]*w[..., None]

    w_final = t_cum
    color = color + color_bg*w_final[..., None]

    return color




        
