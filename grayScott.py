from operator import neg
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy as sp
from torch.autograd import grad

from torchvision.utils import make_grid


# from moviepy.editor import VideoClip
# from moviepy.video.io.bindings import mplfig_to_npimage

device = 'cuda'

class multiplicativeLayer(nn.Module):
    def __init__(self, nc, nhid, nprod=3):
        super(multiplicativeLayer, self).__init__()
        self.nc = nc
        self.nhid = nhid
        self.nprod = nprod

        self.K = nn.ModuleList()
        for i in range(nprod):
            Ki = torch.nn.Conv2d(nc, nhid, 1, 1)
            self.K.append(Ki)
        self.Kclose = torch.nn.Conv2d(nhid, nc, 1, 1)

    def forward(self, U):
        z = 1.0
        for i in range(self.nprod):
            zi = self.K[i](U)
            zi = F.instance_norm(zi)
            zi = F.leaky_relu(zi, negative_slope=0.2)
            z = z*zi
        z = self.Kclose(z)
        return z


class reaction(nn.Module):
    def __init__(self, nhid=512):
        super(reaction, self).__init__()
        self.nhid = nhid

        self.K1 = nn.Parameter(torch.randn(nhid, 2, 1, 1))
        self.K2 = nn.Parameter(torch.randn(nhid, nhid, 1, 1))
        self.K3 = nn.Parameter(torch.randn(2, nhid, 1, 1))

    def forward(self, U):
        U = F.conv2d(U, self.K1)
        U = F.instance_norm(U)
        U = F.leaky_relu(U, negative_slope=0.2)
        U = F.conv2d(U, self.K2)
        U = F.instance_norm(U)
        U = F.leaky_relu(U, negative_slope=0.2)
        U = F.conv2d(U, self.K3)

        return U


class reactionGS(nn.Module):
    def __init__(self, theta=torch.tensor([0.035, 0.065])):
        super(reactionGS, self).__init__()
        # self.theta = torch.tensor([0.060, 0.062])  # pattern
        # self.theta = torch.tensor([0.035, 0.065]) # dots

        self.theta = nn.Parameter(theta)

    def forward(self, U):
        c = 128 ** 2
        F = self.theta[0]
        K = self.theta[1]
        uvv = U[:, 0, :, :] * (U[:, 1, :, :] ** 2)
        du = - uvv + F * (1 - U[:, 0, :, :])
        dv = uvv - (F + K) * U[:, 1, :, :]
        R = torch.zeros_like(U)
        R[:, 0, :, :] = du
        R[:, 1, :, :] = dv

        return c * R

class reactionMul(nn.Module):
    def __init__(self, nhid=512):
        super(reactionMul, self).__init__()
        self.nhid = nhid

        self.K1 = nn.Parameter(torch.randn(nhid, 2, 1, 1))
        self.K2 = nn.Parameter(torch.randn(nhid, nhid, 1, 1))
        self.K3 = nn.Parameter(torch.randn(2, nhid, 1, 1))

        self.MulLayer = multiplicativeLayer(2, nhid, 3)

    def forward(self, U):
        V = self.MulLayer(U)
        U = F.conv2d(U, self.K1)
        U = F.instance_norm(U)
        U = F.leaky_relu(U, negative_slope=0.2)
        U = F.conv2d(U, self.K2)
        U = F.instance_norm(U)
        U = F.leaky_relu(U, negative_slope=0.2)
        U = F.conv2d(U, self.K3)

        return U + V



class GrayScott(nn.Module):
    """Class to solve Gray-Scott Reaction-Diffusion equation"""

    def __init__(self, reaction, N=256, device='cuda'):
        super(GrayScott, self).__init__()
        self.N = N
        self.h = 1 / N
        self.dt = 1 / (N ** 2)

        self.D = torch.tensor([0.14, 0.06], device=device)
        r = 16
        self.U = torch.zeros(1, 2, N, N, device=device)
        self.U[:, 0, :, :] = 1.0 + 0.02 * torch.randn((N, N), device=device)
        self.U[:, 1, :, :] = 0.02 * torch.randn((N, N), device=device)

        N2 = N // 2
        self.U[:, 0, N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
        self.U[:, 1, :, :] = 1 - self.U[:, 0, :, :]

        self.reaction = reaction

    def divGrad(self, u):
        # pad u
        U = torch.zeros((u.shape[0], u.shape[1], u.shape[2] + 2, u.shape[3] + 2), device=u.device)
        U[:, :, 1:-1, 1:-1] = u
        U[:, :, 1:-1, 0] = u[:, :, :, 0]
        U[:, :, 1:-1, -1] = u[:, :, :, -1]
        U[:, :, 0, 1:-1] = u[:, :, 0, :]
        U[:, :, -1, 1:-1] = u[:, :, -1, :]

        v = -4 * U[:, :, 1:-1, 1:-1] + U[:, :, 2:, 1:-1] + U[:, :, :-2, 1:-1] + U[:, :, 1:-1, 2:] + U[:, :, 1:-1, :-2]
        return v / (self.h ** 2)

    def forward(self, Nt):
        """Integrate the resulting system of equations using the Euler method"""
        U = self.U

        UU = torch.zeros(Nt + 1, 2, self.N, self.N, device=U.device)
        UU[0, :, :, :] = U

        # evolve in time using Euler method
        for i in range(int(Nt)):

            LapU = self.divGrad(U)
            DLapU = self.D[None, :, None, None] * LapU

            R = self.reaction(U)

            U = U + self.dt * (DLapU + R)
            UU[i + 1, :, :, :] = U

            if i % 1000 == 0:
                print('time = %3d' % (i))

        return UU



####   Generate the data
N = 128  ## grid size (change to smaller number for faster simulations)
Nt = 1500  ## duraction of simulation

# reactNet = reaction(nhid)
reactNet = reactionGS().to(device)
net = GrayScott(reactNet, N).to(device)


U = net(Nt)
print('done data prep')

# fig, ax = plt.subplots()
# for i in range(U.shape[0] // 10):
#     ax.clear()
#     ax.imshow(U[10 * i, 0, :, :].detach())
#     ax.set_title('time = %3d' % (i))
#     plt.pause(0.01)

# Process the data for learning
Umid = 0.5 * (U[1:] + U[:-1])
dUdt = (U[1:] - U[:-1]) / net.dt
LapU = net.divGrad(Umid)

Rtrue = dUdt - net.D[None, :, None, None] * LapU
Rtrue = Rtrue.detach()
Umid  = Umid.detach()
print('done data proc')


### Learn the reaction term
nhid = 512
reactNetNN = reaction(nhid).to(device)
#reactNetNN = reactionMul(nhid).to(device)

lr =  1e-3
optimizer = optim.Adam(reactNetNN.parameters(), lr=lr)

epochs = 1000
batchsize = 50
nbatches = Rtrue.shape[0]//batchsize

for i in range(epochs):
    avloss = 0
    perm = torch.randperm(Rtrue.shape[0])
    for j in range(nbatches):
        batch = perm[j*batchsize:(j+1)*batchsize]
        optimizer.zero_grad()

        R = reactNetNN(Umid[batch])
        loss = F.mse_loss(R, Rtrue[batch])/F.mse_loss(Rtrue[batch]*0, Rtrue[batch])
        avloss += loss.item()
        loss.backward()
        optimizer.step()

        #print('batchid = %3d   loss = %3.2e'%(j, loss))

    print('===== epoch = %3d   avloss = %3.2e ========'%(i, avloss/nbatches))

print('done')


inference = False
if inference:
    reactNetNN = reactNetNN.to('cpu')
    trainedNet = GrayScott(reactNetNN, N, device='cpu').to('cpu')
    with torch.no_grad():
        U = trainedNet(Nt)

        fig, ax = plt.subplots()
        for i in range(U.shape[0] // 10):
            ax.clear()
            ax.imshow(U[10 * i, 0, :, :].detach().cpu())
            ax.set_title('time = %3d' % (i))
            plt.pause(0.1)


