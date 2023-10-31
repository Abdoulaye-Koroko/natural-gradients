import torch
import torch.nn as nn
from torchvision import datasets,transforms
import numpy as np
import argparse
import os

from optimizers.kfac import KFAC
from optimizers.kpsvd import KPSVD
from optimizers.deflation import Deflation
from optimizers.kfac_cor import KFAC_CORRECTED
from optimizers.lanczos import Lanczos
from optimizers.twolevel_kfac import TwolevelKFAC
from optimizers.exact_natural_gradient import ExactNG


class Mymodel(nn.Module):
    
    def __init__(self,num_classes=10):
        super(Mymodel,self).__init__()
        
        self.num_classes = num_classes
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
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
    

def train(args):

    #device
    device = torch.device("cuda") # or torch.device("cpu")

    # Define your model
    model = Mymodel(num_classes=10)

    model = model.to(device)

    #Define the optimizer (natural gradient-based method)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,nesterov=False,weight_decay=args.weight_decay)

    preconditioner = KFAC(model,damping=args.damping) # It can be any of the imported preconditioner above (e.g. KPSVD, Deflation, etc.)

    #Define your dataloader
    transform = transforms.Compose(
        [transforms.ToTensor()])


    dataset = datasets.MNIST(root="./data/", train=True,
                                        download=True,transform=transform)

    trainloader =  torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=True,drop_last=True)
    #Define your loss function
    criterion =  nn.CrossEntropyLoss() 
    
    losses = []
    
    best_loss = np.inf

    #Training loop
    for epoch in range(args.num_epochs):
        
        train_loss = 0

        model.train()

        for iter,batch in enumerate(trainloader):

            optimizer.zero_grad()

            inputs,labels = batch

            inputs,labels = inputs.to(device),labels.to(device)
            
            preconditioner.update_stats = True

            with torch.set_grad_enabled(True): # Not obliged

                outputs = model(inputs)

                loss = criterion(outputs,labels)

            loss.backward() #Compute the gradients with targets from training data

            preconditioner.step(update_params=True) # Preconditionnes the gradients with the computed Fisher   

            optimizer.step() # perform an iteration of the optimization
            
            train_loss+=loss.item()*inputs.size(0)
            
        train_loss/=len(dataset)
        
        losses.append(train_loss)
        
        print(f"Epoch {epoch}/{args.num_epochs}:")
        
        print(f"\tTrain loss: {train_loss}")
        
        if train_loss<best_loss:
            
            best_loss = train_loss
        
    results =  dict(train_loss=torch.tensor(losses,device="cpu"),epochs = np.arange(1,args.num_epochs))
    
    return best_loss, results


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Function arguments')
   
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    
    parser.add_argument('--result_name', type=str, default="kfac",
                        help='tensorbord_name name')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    
    
    parser.add_argument('--damping', type=float, default=1e-4,
                        help='damping')
      
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum parameter')
    
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='weight_decay parameter')
    

    args = parser.parse_args()
    
    _,results = train(args)
    
    output_folder = "examples/results/"
    
    if not os.path.exists(output_folder):
        
        os.makedirs(output_folder)
        
    np.save(output_folder + f'{args.result_name}.npy', results) 
    
    print(f"The results are saved in {output_folder} under the name {args.result_name}.npy")
    
    
    # You can use notebook.ipynb to visualize the results.
        
    
