import torch


image_resolution =100
res_squared = image_resolution**2
u= torch.zeros((image_resolution,image_resolution))
v= torch.zeros((image_resolution,image_resolution))

for i in range(image_resolution):
    u[:,i]= i
    v[i,:]=i




def directions(camx,camy,x,y,transform,fx=1,fy=1):
    directions_cam = torch.zeros(size=[res_squared,3])

    directions_cam[0][i]=i
    directions_cam[1][0] = (i-camx)/fx
    directions_cam[1][1]= (i-camy)/fy
    directions_cam[1][2]= -1
    rot = torch.zeros((3,3))
    rot[0] = transform[0,0:3]
    rot[1] = transform[1,0:3]
    rot[2] = transform[2,0:3]

    trans = torch.zeros((1,3))
    trans[0] = transform[0,3]
    trans[1] = transform[1,3]
    trans[2] = transform[2,3]

    directions_world = directions_cam@rot + trans

    return directions_world






    

