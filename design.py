# Deep Convolutional GANs

# Importing the libraries
# from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import numpy as np
from mpi4py import MPI


# Creating the network to create the peer2peer connection for swaping of the Discriminator

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
nc = 3

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) 
# We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 


""" 
We use dataLoader to get the images of the training set batch by batch.
We ust the shuffle = True because we want to get the dataset in random order so that we can train model more precisely.
We use num_worker = 2 which represent the number of thread and the worker servers to define the 
"""

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



# Defining the copy of the generator to shuffle between diffrent severs
def copyGenerator():
    layer_num = 0
    for param in netG.parameters():
        #print(rank, "started")
        if (rank == 0):
            data = param.data.numpy().copy()
            #print(rank, data.shape)
        else:
            data = None
            #print(rank, data.shape)

        #print(rank, "before bcast")
        #comm.Barrier()
        data = comm.bcast(data, root = 0)
        #print(rank, "after bcast")
        if (rank != 0):
            param.data = torch.from_numpy(data)
            #print("Node rank " + str(rank) + " has synched generator layer " + str(layer_num))

        layer_num += 1
        #comm.Barrier()


#Peer2Peer shuffling of the Discriminator
def shuffleDiscriminators():
    if (rank != 0):
        layer_num = 0
        for param in netD.parameters():
            outdata = param.data.numpy().copy()
            indata = None

            if (rank != size - 1):
                comm.send(outdata, dest=rank + 1, tag=1)
            if (rank != 1):
                indata = comm.recv(source = rank-1, tag=1)

            if (rank == size - 1):
                comm.send(outdata, dest=1, tag=2)
            if (rank == 1):
                indata = comm.recv(source = size - 1, tag=2)
            # Shuffling the Discriminator
            param.data = torch.from_numpy(indata)
            layer_num += 1



# Defining the generator

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


# Creating the generator
netG = G()
netG.apply(weights_init)

# Defining the discriminator

class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

# Creating the discriminator
netD = D()
netD.apply(weights_init)



# Training the DCGANs

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

for epoch in range(3):

    if (epoch % 2 == 0):
        shuffleDiscriminators()

    for i, data in enumerate(dataloader, 775):
        
        # 1st Step: Updating the weights of the neural network of the discriminator

        netD.zero_grad()
        
        # Training the discriminator with a real image of the dataset
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)
        
        # Training the discriminator with a fake image generated by the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)
        
        # Backpropagating the total error
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # 2nd Step: Updating the weights of the neural network of the generator

        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()
        
        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.item(), errG.item()))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)