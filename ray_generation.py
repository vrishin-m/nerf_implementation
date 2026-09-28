import torch
import random



def generate_matrix(sample, image_resolution = 100, batch_size = 1024):
    pose = sample[pose]

    camera_position = pose[:3, 3]
    camx,camy = camera_position[0], camera_position[1]
    pixel = torch.zeros([batch_size,2])
    rays_direction = torch.zeros([batch_size,3])
    for i in range(batch_size):
        x,y = random.randint(0,image_resolution), random.randint(0,image_resolution)
        pixel[i]= torch.tensor([x,y])
        rays_direction[i]= directions(camx,camy,x,y,pose)

    return pixel, rays_direction, camera_position

  
    




def directions(camx,camy,x,y,transform,fx=1,fy=1):
    directions_cam = torch.zeros(size=[3])

    directions_cam[1][0] = (x-camx)/fx
    directions_cam[1][1]= (y-camy)/fy
    directions_cam[1][2]= -1
    rot = torch.zeros((3,3))
    rot[0] = transform[0,0:3]
    rot[1] = transform[1,0:3]
    rot[2] = transform[2,0:3]

    trans = torch.zeros((1,3))
    trans[0:3] = transform[0:3,3]


    directions_world = directions_cam@rot + trans

    return directions_world




def march_rays(sample_image, near_bound = 2, far_bound = 6, samples= 64):
    pixel, rays_direction, camera_position = generate_matrix(sample_image)
    t= torch.linspace(near_bound,far_bound, 64)
    points = torch.tensor([camera_position + rays_direction*i for i in t])
    return points

    

