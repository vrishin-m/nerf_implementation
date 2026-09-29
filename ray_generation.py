import torch
import random
import math

#march_rays is the "main" func of this program

def generate_matrix(sample,batch_size, camera_angle_x, image_resolution = 100):
    pose = sample["pose"]

    camera_position = pose[:3, 3]
    pixel = torch.zeros([batch_size,2])
    rays_direction = torch.zeros([batch_size,3])
    for i in range(batch_size):
        x,y = random.randint(0,image_resolution-1), random.randint(0,image_resolution-1)
        pixel[i]= torch.tensor([x,y])
        rays_direction[i]= directions(x,y,pose,camera_angle_x,image_resolution)

    return rays_direction, camera_position

  
    




def directions(x,y,transform,camera_angle_x,image_res):
    fx= fy = image_res/(2 * math.tan(camera_angle_x/2))
    directions_cam = torch.zeros(size=[3])

    directions_cam[0] = (x-image_res/2)/fx
    directions_cam[1]= -(y-image_res/2)/fy
    directions_cam[2]= -1
    rot = torch.zeros((3,3))
    rot[0] = transform[0,0:3]
    rot[1] = transform[1,0:3]
    rot[2] = transform[2,0:3]



    directions_world = directions_cam@rot 
    directions_world= directions_world / (torch.linalg.vector_norm(directions_world,2))
    return directions_world




def march_rays(sample_image, camera_angle_x, near_bound = 2, far_bound = 6, samples= 64, batch_size = 1024):
    rays_direction, camera_position = generate_matrix(sample_image, batch_size, camera_angle_x)
    cam_pos_tensor = camera_position.unsqueeze(0).expand(batch_size,-1)
    t= torch.linspace(near_bound,far_bound, samples)
    cam_pos_tensor= cam_pos_tensor.unsqueeze(1)
    rays_direction=rays_direction.unsqueeze(1)
    t= t.unsqueeze(0).unsqueeze(2)
    points = cam_pos_tensor + rays_direction*t
    delta = (far_bound-near_bound)/samples
    return points, sample_image["pose"][:3,:3], delta

    

