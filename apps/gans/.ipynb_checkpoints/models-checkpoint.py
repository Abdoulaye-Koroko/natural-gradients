import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import grad, Variable


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        
        super(Generator, self).__init__()
        
        nz = z_dim
        
        nc = im_chan
        
        ngf = hidden_dim
        
        self.z_dim = z_dim
        
        
        self.gen = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

        
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

    
def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)






class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        
        super(Critic, self).__init__()
        
        nc = im_chan
        
        ndf = hidden_dim
        
        self.crit = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )
        
    
    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)
    
    
    

class ResBlock(nn.Module):
    
    def __init__(self, num_filters, resample=None, batchnorm=True, inplace=False):
        super(ResBlock, self).__init__()

        if resample == 'up':
            conv_list = [nn.ConvTranspose2d(num_filters, num_filters, 4, stride=2, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, padding=1)]
            self.conv_shortcut =  nn.ConvTranspose2d(num_filters, num_filters, 1, stride=2, output_padding=1)

        elif resample == 'down':
            conv_list = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)]
            self.conv_shortcut = nn.Conv2d(num_filters, num_filters, 1, stride=2)

        elif resample == None:
            conv_list = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, padding=1)]
            self.conv_shortcut = None
        else:
            raise ValueError('Invalid resample value.')

        self.block = []
        for conv in conv_list:
            if batchnorm:
                self.block.append(nn.BatchNorm2d(num_filters))
            self.block.append(nn.ReLU(inplace))
            self.block.append(conv)

        self.block = nn.Sequential(*self.block)


    def forward(self, x):
        shortcut = x
        if not self.conv_shortcut is None:
            shortcut = self.conv_shortcut(x)
        return shortcut + self.block(x)

class ResNet32Generator(nn.Module):
    def __init__(self, n_in, n_out, num_filters=128, batchnorm=True):
        super(ResNet32Generator, self).__init__()
        self.num_filters = num_filters

        self.input = nn.Linear(n_in, 4*4*num_filters)
        self.network = [ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=True),
                        ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=True),
                        ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=True)]
        if batchnorm:
            self.network.append(nn.BatchNorm2d(num_filters))
        self.network += [nn.ReLU(True),
                        nn.Conv2d(num_filters, 3, 3, padding=1),
                        nn.Tanh()]

        self.network = nn.Sequential(*self.network)

    def forward(self, z):
        x = self.input(z).view(len(z), self.num_filters, 4, 4)
        return self.network(x)

class ResNet32Discriminator(nn.Module):
    def __init__(self, n_in, n_out, num_filters=128, batchnorm=False):
        super(ResNet32Discriminator, self).__init__()

        self.block1 = nn.Sequential(nn.Conv2d(3, num_filters, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1))

        self.shortcut1 = nn.Conv2d(3, num_filters, 1, stride=2)

        self.network = nn.Sequential(ResBlock(num_filters, resample='down', batchnorm=batchnorm),
                                    ResBlock(num_filters, resample=None, batchnorm=batchnorm),
                                    ResBlock(num_filters, resample=None, batchnorm=batchnorm),
                                    nn.ReLU())
        self.output = nn.Linear(num_filters, 1)

    def forward(self, x):
        y = self.block1(x)
        y = self.shortcut1(x) + y
        y = self.network(y).mean(-1).mean(-1)
        y = self.output(y)

        return y
    
