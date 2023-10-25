import torch.nn as nn
from torchvision import models
import torch

class cnn_autoencoder_mnist(nn.Module):

    def __init__(self):
        super(cnn_autoencoder_mnist,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(in_channels=16,out_channels=6,kernel_size=5),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(in_channels=6,out_channels=1,kernel_size=5),
            nn.Sigmoid()
            
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
class cnn_autoencoder_cifar(nn.Module):

    def __init__(self):
        super(cnn_autoencoder_cifar,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=6),
            
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=16),)

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(in_channels=16,out_channels=6,kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=6),
            
            nn.ConvTranspose2d(in_channels=6,out_channels=3,kernel_size=5),
            
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class cnn_autoencoder_svhn(nn.Module):

    def __init__(self):
        super(cnn_autoencoder_svhn,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=6),
            
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=16))

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(in_channels=16,out_channels=6,kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=6),
            
            nn.ConvTranspose2d(in_channels=6,out_channels=3,kernel_size=5),
            
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

