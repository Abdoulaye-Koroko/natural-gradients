import argparse
import os
import itertools
import time
import numpy as np
import torch

from apps.cnn_autoencoder.train import train


parser = argparse.ArgumentParser(description='Function arguments')
    
parser.add_argument('--optim', type=str, default="kfac",
                    help='optimizer name')

parser.add_argument('--num_epochs', type=int, default=5,
                    help='number of epochs')

parser.add_argument('--data', type=str, default="MNIST",
                    help='data name')

parser.add_argument('--data_path', type=str, default="./data/",
                    help='path towards the folder containing the data')

parser.add_argument('--result_name', type=str, default="kfac",
                    help='tensorbord_name name')

parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size')

parser.add_argument('--lr', type=float, default=1e-2,
                    help='Learning rate')


parser.add_argument('--damping', type=float, default=1e-4,
                    help='damping')

parser.add_argument('--clipping', type=float, default=1e-3,
                    help='kl clipping')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum parameter')

parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight_decay parameter')

args = parser.parse_args()


def grid_search(args):
    
    t = time.time()
    
    best_loss = 100
    
    best_params = None
    
    best_result = None
    
    root = "apps/cnn_autoencoder/results/"
    
    if not os.path.exists(root):
        
        os.makedirs(root)
    
    name = args.result_name
    
    npy_dir = os.path.join(root,name)

    if args.optim == "kfac":
        
        lrs = [1e-2]
        
        dampings = [1e-2]
        
        weight_decays = [1e-3]
        
        momentums = [0.9]
        
        list_params = [lrs,dampings,weight_decays,momentums]

        for params in itertools.product(*list_params):
            
            lr,damp,wd,momentum = params 
            
            args.lr = lr
            
            args.damping = damp
            
            args.weight_decay = wd
            
            args.momentum = momentum
            
            print(f"Processing for params (lr,damp,weight_decay,momentum): {params}")
            
            print(20*"+++++")
            
            print("\n")
            
            #try:
            loss,result = train(args)
            
            if loss<best_loss:
                
                best_loss = loss
                
                best_params = {"lr":lr,"damping":damp,"weigt_decay":wd,"momentum":momentum}
                
                best_result = result


            #except:
             #   continue

                    
    elif args.optim in ["sgd","adam"]:
        
        lrs = [1e-2]
        
        weight_decays = [1e-3]
        
        momentums = [0.9]
        
        list_params = [lrs,weight_decays,momentums]
        
         
        for params in itertools.product(*list_params):
            
            lr,wd,momentum = params 
            
            args.lr = lr
            
            args.weight_decay = wd
            
            args.momentum = momentum
            
            print(f"Processing for params: {params}")
            
            print(20*"+++++")
            
            print("\n")
            
            loss,result = train(args)
            
            if loss<best_loss:
                
                best_loss = loss
                
                best_params = {"lr":lr,"weigt_decay":wd,"momentum":momentum}
                
                best_result = result
        
    print(f"Best params: {best_params}")
    
    print(f"Best loss: {best_loss}")
    
    np.save(npy_dir+'_params.npy',best_params)
    
    np.save(npy_dir+'.npy', best_result)
    
    time_elapsed = time.time()-t
    
    print(f"\tProcessing time: {time_elapsed // 60} minutes {time_elapsed % 60} seconds")
    




if __name__ == '__main__':
    
    
        
    name = args.optim

    print(f"Processing for optimizer: {args.optim}")

    print(50*"===")

    args.result_name = name

    grid_search(args)


