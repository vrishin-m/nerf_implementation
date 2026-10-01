import torch
import math

def render_ray(sigma, rgb, color_bg = torch.zeros([3]), delta=1/16, samples=64):
    t_cum=1
    color = torch.zeros([3])
    for i in range(samples):
        t= math.exp(-sigma[i]*delta)
        alpha = 1-t
        w = t_cum * alpha
        t_cum *= t
        color += rgb[i]*w

    w_final = t_final = t_cum* (1-alpha)
    color += color_bg*w_final

    return color




        
