import torch.nn as nn
from torchvision import models
import torch

seed=0

#MLP for testing two-level KFAC

class DeepLinear(nn.Module):
    
    def __init__(self):
        super(DeepLinear,self).__init__()
        
        self.model = [nn.Linear(28*28,10),nn.BatchNorm1d(num_features=10)]
        
        for i in range(62):
            self.model.append(nn.Linear(10,10))
            self.model.append(nn.BatchNorm1d(num_features=10))
            #self.model.append(nn.ReLU())
            
        self.model.append(nn.Linear(10,10))
        
        self.model = nn.Sequential(*self.model)
    
    def forward(self,x):
        return self.model(x)



class ToyNet(nn.Module):
    def __init__(self,num_classes=10):
        
        super(ToyNet, self).__init__()
        
        self.model = nn.Sequential(
        nn.Linear(28*28, 10,bias=True),
        nn.Tanh(),
        
        nn.Linear(10, 30,bias=True),
        nn.Tanh(),    
        
        nn.Linear(30,20,bias=True),
        nn.Tanh(),
            
        nn.Linear(20,15,bias=True),
        nn.Tanh(),
            
        nn.Linear(15, 10,bias=True),
         
        
        )
        
    def forward(self,x):
        return self.model(x)
        
    
    
# MLP
class MLP(nn.Module):
    
    def __init__(self,input_dim,num_classes=10):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        
        self.model = nn.Sequential(
            
        nn.Linear(self.input_dim, 1000,bias=True),
        nn.BatchNorm1d(num_features=1000),
        
        
        nn.Linear(1000, 900, bias=True),
        nn.BatchNorm1d(num_features=900),
        

        nn.Linear(900,800,bias=True),
        nn.BatchNorm1d(num_features=800),
        
   
        nn.Linear(800,700,bias=True),
        nn.BatchNorm1d(num_features=700),
        
        
        nn.Linear(700,600,bias=True),
        nn.BatchNorm1d(num_features=600),
        
        
        nn.Linear(600,500,bias=True),
        nn.BatchNorm1d(num_features=500),
        
        
        nn.Linear(500,400,bias=True),
        nn.BatchNorm1d(num_features=400),
        
        
        nn.Linear(400,300,bias=True),
        nn.BatchNorm1d(num_features=300),
        
        
        nn.Linear(300,200,bias=True),
        nn.BatchNorm1d(num_features=200),
        
        
        nn.Linear(200,100,bias=True),
        nn.BatchNorm1d(num_features=100),
        
            
        nn.Linear(100,50,bias=True),
        nn.BatchNorm1d(num_features=50),
           
        
        nn.Linear(50,20,bias=True),
        nn.BatchNorm1d(num_features=20),
            
        
        
        nn.Linear(20,self.num_classes,bias=True),
        )
        
    
        
    def forward(self,x):
            
        x = self.model(x)
            
        return x


# Deep autoencoder MNIST
class autoencoder_mnist(nn.Module):
    
    def __init__(self):
        super(autoencoder_mnist, self).__init__()
        
        self.input_dim = 784
        
        
        self.encoder = nn.Sequential(
        nn.Linear(self.input_dim, 1000,bias=True),
        nn.ReLU(),
        
        nn.Linear(1000, 500, bias=True),
        nn.ReLU(),

        nn.Linear(500,250,bias=True),
        nn.ReLU(),
   
        nn.Linear(250,30,bias=True),
        nn.ReLU()
        )
        
        
        
        self.decoder = nn.Sequential(
        nn.Linear(30, 250,bias=True),
        nn.ReLU(),
        
        nn.Linear(250, 500, bias=True),
        nn.ReLU(),
        

        nn.Linear(500,1000,bias=True),
        nn.ReLU(),
        
   
        nn.Linear(1000,self.input_dim,bias=True),
	    nn.Sigmoid(),
	
        )
        
        
    def forward(self,x):
            
        x = self.encoder(x)
            
        x = self.decoder(x)
            
        return x
    


# Deep autoencoder FACES
class autoencoder_faces(nn.Module):
    
    def __init__(self):
        super(autoencoder_faces, self).__init__()
        
        self.input_dim = 625
        
        
        self.encoder = nn.Sequential(
        nn.Linear(self.input_dim, 2000,bias=True),
        nn.ReLU(),
        
        nn.Linear(2000, 1000, bias=True),
        nn.ReLU(),
        

        nn.Linear(1000,500,bias=True),
        nn.ReLU(),
   
        nn.Linear(500,30,bias=True),
        nn.ReLU()
        
        )
        
        
        
        self.decoder = nn.Sequential(
        nn.Linear(30, 500,bias=True),
        nn.ReLU(),
        
        
        nn.Linear(500, 1000, bias=True),
        nn.ReLU(),
        

        nn.Linear(1000,2000,bias=True),
        nn.ReLU(),
        
   
        nn.Linear(2000,self.input_dim,bias=True)
        )
        
        
    def forward(self,x):
            
        x = self.encoder(x)
            
        x = self.decoder(x)
            
        return x
    

    
# Deep autoencoder CURVES
class autoencoder_curves(nn.Module):
    
    def __init__(self):
        super(autoencoder_curves, self).__init__()
        
        self.input_dim = 784
        
        
        self.encoder = nn.Sequential(
        nn.Linear(self.input_dim, 400,bias=True),
        nn.ReLU(),
        
        nn.Linear(400, 200, bias=True),
        nn.ReLU(),

        nn.Linear(200,100,bias=True),
        nn.ReLU(),
   
        nn.Linear(100,50,bias=True),
        nn.ReLU(),
        
        nn.Linear(50,25,bias=True),
        nn.ReLU(),
        
        nn.Linear(25,6,bias=True),
        nn.ReLU(),
        )
        
        
        
        self.decoder = nn.Sequential(
        nn.Linear(6, 25,bias=True),
        nn.ReLU(),
        
        nn.Linear(25, 50, bias=True),
        nn.ReLU(),

        nn.Linear(50,100,bias=True),
        nn.ReLU(),
    
        
        nn.Linear(100,200,bias=True),
        nn.ReLU(),
        
        
        nn.Linear(200,400,bias=True),
        nn.ReLU(),
        
   
        nn.Linear(400,self.input_dim,bias=True),
        nn.Sigmoid(),
        )
        
        
    def forward(self,x):
            
        x = self.encoder(x)
            
        x = self.decoder(x)
            
        return x
    

# Deep autoencoder CIFAR10
class autoencoder_cifar10(nn.Module):
    
    def __init__(self):
        super(autoencoder_cifar10, self).__init__()
        
        self.input_dim = 32*32*3
        
        
        self.encoder = nn.Sequential(
        nn.Linear(self.input_dim, 2000,bias=True),
        nn.ReLU(),
        
        nn.Linear(2000, 1000, bias=True),
        nn.ReLU(),

        nn.Linear(1000,500,bias=True),
        nn.ReLU(),
   
        nn.Linear(500,30,bias=True),
        nn.ReLU()
        )
        
        
        
        self.decoder = nn.Sequential(
        nn.Linear(30, 500,bias=True),
        nn.ReLU(),
        
        nn.Linear(500, 1000, bias=True),
        nn.ReLU(),

        nn.Linear(1000,2000,bias=True),
        nn.ReLU(),
   
        nn.Linear(2000,self.input_dim,bias=True)
        )
        
        
    def forward(self,x):
            
        x = self.encoder(x)
            
        x = self.decoder(x)
            
        return x
    

#Cuda-convnet
class cuda_convnet(nn.Module):
    
    def __init__(self,num_classes=10):
        super(cuda_convnet,self).__init__()
        self.num_classes = num_classes
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=64,out_features = self.num_classes)
            
        )
    def forward(self,x):
        x = self.model(x)
        return x


class ToyCNN(nn.Module):
    
    def __init__(self,num_classes=10):
        super(ToyCNN,self).__init__()
        self.num_classes = num_classes
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=4,kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3),
            
            nn.Conv2d(in_channels=4,out_channels=2,kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(),
            
            nn.Linear(in_features=2,out_features = self.num_classes)
            
        )
    def forward(self,x):
        x = self.model(x)
        #print(f"size: {x.size()}")
        return x
    
    

# Resnet 18
def resnet18(num_classes):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model  





# Resnet 34
def resnet34(num_classes):
    model = models.resnet34(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model 


#VGG 11
def vgg11(num_classes):
    model = models.vgg11(pretrained=False)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return model


def resnet50(num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model 
