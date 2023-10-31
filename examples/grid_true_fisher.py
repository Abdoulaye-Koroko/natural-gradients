import argparse
import os
import itertools
import time
import numpy as np
import torch

from examples.train_empirical_fisher import train


parser = argparse.ArgumentParser(description='Function arguments')
   
parser.add_argument('--num_epochs', type=int, default=5,
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
    

def grid_search(args):
    
    t = time.time()
    
    best_loss = 100
    
    best_params = None
    
    best_result = None
    
    root = "examples/results/"
    
    if not os.path.exists(root):
        
        os.makedirs(root)
    
    name = args.result_name
    
    npy_dir = os.path.join(root,name)

   
    # Define your hyper-parameters search space
        
    lrs = [1e-5]

    dampings = [1]

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

        try:
            loss,result = train(args)

            if loss<best_loss:

                best_loss = loss

                best_params = {"lr":lr,"damping":damp,"weigt_decay":wd,"momentum":momentum}

                best_result = result


        except:

            continue

    print(f"Best params: {best_params}")
    
    print(f"Best loss: {best_loss}")
    
    np.save(npy_dir+'_params.npy',best_params)
    
    np.save(npy_dir+'.npy', best_result)
    
    time_elapsed = time.time()-t
    
    print(f"\tProcessing time: {time_elapsed // 60} minutes {time_elapsed % 60} seconds")
    


if __name__ == '__main__':

    print(f"Grid search for {args.result_name}  optimizer")

    print(50*"===")

    grid_search(args)
            
