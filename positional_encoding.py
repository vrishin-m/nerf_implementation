import math
import torch

L_pos = 10 
L_dir = 4
#position will be size 60, direction 16


def encode(p):
    gammax,gammay,gammaz, gammatheta, gammaphi=[],[],[],[],[]
    for i in range(L_pos):
        gammax.append(math.sin(2**i*math.pi*float(p[0])))
        gammax.append( math.cos(2**i*math.pi*p[0]))
        gammay.append( math.sin(2**i*math.pi*p[1]))
        gammay.append( math.cos(2**i*math.pi*p[1]))
        gammaz.append( math.sin(2**i*math.pi*p[2]) )
        gammaz.append( math.cos(2**i*math.pi*p[2]))

    for i in range(L_dir):
        gammatheta.append(math.sin(2**i*math.pi*float(p[3])))
        gammatheta.append( math.cos(2**i*math.pi*p[3]))
        gammaphi.append( math.sin(2**i*math.pi*p[4]))
        gammaphi.append( math.cos(2**i*math.pi*p[4]))

    gammapos = gammax+gammay+gammaz
    gammadir= gammatheta+gammaphi
    return torch.tensor(gammapos), torch.tensor(gammadir)

    
        
p= torch.tensor([1,2,3,4,5])
print(encode(p))