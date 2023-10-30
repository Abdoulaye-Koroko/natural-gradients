import argparse
import os
import itertools
import time
import numpy as np
import torch

from train_autoencoder import train_autoencoder
from train_cnn import train_cnn



"""
auto_parser = argparse.ArgumentParser(description='Function arguments')
    
auto_parser.add_argument('--optim', type=str, default="kfac",
                    help='optimizer name')

auto_parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')

auto_parser.add_argument('--device', type=str, default="cuda:0",
                    help='device')


auto_parser.add_argument('--data', type=str, default="CURVES",
                    help='data name')

auto_parser.add_argument('--tensorboard_name', type=str, default="kfac",
                    help='tensorbord_name name')

auto_parser.add_argument('--two_level', type=int, default=0,
                    help='Use or not two level preconditionning')

auto_parser.add_argument('--dd', type=int, default=1,
                    help=' if True use two level condining as simplified in domain decomposition \
                    if no use the original expression')

auto_parser.add_argument('--batch_size', type=int, default=128,
                    help='batch_size')

auto_parser.add_argument('--fisher_batch', type=int, default=64,
                    help='batch_size used te estimate the Fisher')


auto_parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')


auto_parser.add_argument('--damping', type=float, default=1e-3,
                    help='damping')

auto_parser.add_argument('--kl', type=float, default=1e-2,
                    help='kl clipping')

auto_parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum parameter')

auto_parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight_decay parameter')


auto_parser.add_argument('--root', type=str, default="CURVES",
                    help='root to save results')


auto_parser.add_argument('--train_function', type=str, default="auto",
                    help='Training function')
auto_parser.add_argument('--two_level_coeff', type=float, default=1e-1,
                        help='Two level coefficient parameter')

auto_parser.add_argument('--init_params', type=str, default='xavier_uniform',
                        help='Two level coefficient parameter')

auto_parser.add_argument('--R0', type=str, default='ones',
                        help='How to build colums of R0')

auto_parser.add_argument('--krylov', type=int, default=0,
                    help='Use or not krylov')

auto_args = auto_parser.parse_args()
auto_args.root = auto_args.data

"""

cnn_parser = argparse.ArgumentParser(description='Function arguments')

cnn_parser.add_argument('--model', type=str, default ="cuda_convnet",
                    help='model name')
cnn_parser.add_argument('--optim', type=str, default="kfac",
                    help='optimizer name')

cnn_parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs')

cnn_parser.add_argument('--device', type=str, default="cuda:0",
                    help='device')


cnn_parser.add_argument('--data', type=str, default="CIFAR10",
                    help='data name')

cnn_parser.add_argument('--tensorboard_name', type=str, default="kfac",
                    help='tensorbord_name name')

cnn_parser.add_argument('--two_level', type=int, default=0,
                    help='Use or not two level preconditionning')

cnn_parser.add_argument('--dd', type=int, default=0,
                    help=' if True use two level condining as simplified in domain decomposition \
                    if no use the original expression')

cnn_parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size')


cnn_parser.add_argument('--fisher_batch', type=int, default=64,
                    help='batch_size used te estimate the Fisher')

cnn_parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')


cnn_parser.add_argument('--damping', type=float, default=1e-3,
                    help='damping')

cnn_parser.add_argument('--kl', type=float, default=1e-3,
                    help='kl clipping')

cnn_parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum parameter')

cnn_parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight_decay parameter')


cnn_parser.add_argument('--root', type=str, default ="cuda_convnet",
                    help='results to save data')


cnn_parser.add_argument('--train_function',type = str, default = "cnn",
                    help='training_function')

cnn_parser.add_argument('--two_level_coeff', type=float, default=1e-1,
                        help='Two level coefficient parameter')


cnn_parser.add_argument('--init_params', type=str, default='xavier_uniform',
                        help='Two level coefficient parameter')

cnn_parser.add_argument('--R0', type=str, default='ones',
                        help='How to build colums of R0')

cnn_parser.add_argument('--krylov', type=int, default=0,
                    help='Use or not krylov')


cnn_args = cnn_parser.parse_args()


cnn_args.root = cnn_args.model






def grid_search(args):
    
    if args.train_function=="cnn":
        train_function = train_cnn
    elif args.train_function=="auto":
        train_function = train_autoencoder
    
    t = time.time()
    best_loss = 100
    best_params = None
    best_result = None
    root = args.root
    name = args.tensorboard_name
    npy_dir = os.path.join(root,f"outputs_{args.batch_size}",tensorboard_name)

    #npy_dir = os.path.join("test",root,"outputs",tensorboard_name)

    
    
    if args.optim in ["kfac","kpsvd","kfac_cor","lzs","kpsvd_d"]:
        if not args.two_level or args.dd:
            lrs = [1e-1,1e-2,1e-3,1e-4]
            dampings = [1e-1,1e-2,1e-3,1e-4]
            kls = [1]
            weight_decays = [1e-3]
            momentums = [0.9]
            list_params = [lrs,dampings,kls,weight_decays,momentums]

            for params in itertools.product(*list_params):
                lr,damp,kl,wd,momentum = params 
                args.lr = lr
                args.damping = damp
                args.kl = kl
                args.weight_decay = wd
                args.momentum = momentum
                print(f"Processing for params (lr,damp,kl,weight_decay,momentum): {params}")
                print(20*"+++++")
                print("\n")
                index = np.random.randint(low=0, high=4, size=1)
                index = index[0]
                print("index",index)
                device = f"cuda:{index}"
                args.device = device
                try:
                    loss,result = train_function(args)
                    if loss<best_loss:
                        best_loss = loss
                        best_params = {"lr":lr,"damping":damp,"kl":kl,"weigt_decay":wd,"momentum":momentum}
                        best_result = result
                    
            
                except:
                    continue

                    
                    
        else:
            
            lrs =  [1e-2]
            dampings =  [1e-3]
            kls = [1e-2]
            weight_decays = [1e-3]
            momentums = [0.9]
            two_level_coffs = [1e-1]
            list_params = [lrs,dampings,kls,weight_decays,momentums,two_level_coffs]

            for params in itertools.product(*list_params):
                lr,damp,kl,wd,momentum,twc = params 
                args.lr = lr
                args.damping = damp
                args.kl = kl
                args.weight_decay = wd
                args.momentum = momentum
                args.two_level_coeff = twc
                print(f"Processing for params (lr,damp,kl,weight_decay,momentum,two_level_coeff): {params}")
                print(20*"+++++")
                print("\n")
                index = np.random.randint(low=0, high=4, size=1)
                index = index[0]
                print("index",index)
                device = f"cuda:{index}"
                args.device = device
                try:
                    loss,result = train_function(args)
                    if loss<best_loss:
                        best_loss = loss
                        best_params = {"lr":lr,"damping":damp,"kl":kl,"weigt_decay":wd,"momentum":momentum,"two_level_coeff":twc}
                        best_result = result
                except:
                    continue


    elif args.optim in ["sgd","adam"]:
        lrs = [1e-1,1e-2,1e-3,1e-4]
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
            loss,result = train_function(args)
            if loss<best_loss:
                best_loss = loss
                best_params = {"lr":lr,"weigt_decay":wd,"momentum":momentum}
                best_result = result
        
    elif args.optim in ["NG","NG_BD","NG_BD_heuristic","kfac_standard"]:
        
        lrs = [1e-3,1e-4]
        dampings = [1e-2,1e-3,1e-4]
        weight_decays = [1e-3]
        momentums = [0.9]
        list_params = [lrs,dampings,weight_decays,momentums]

        for params in itertools.product(*list_params):
            lr,damp,wd,momentum = params 
            args.lr = lr
            args.damping = damp
            args.weight_decay = wd
            args.momentum = momentum
            print(f"Processing for params (lr,damp,kl,weight_decay,momentum): {params}")
            print(20*"+++++")
            print("\n")
            index = np.random.randint(low=0, high=4, size=1)
            index = index[0]
            print("index",index)
            device = f"cuda:{index}"
            args.device = device
            try:
                loss,result = train_function(args)
                if loss<best_loss:
                    best_loss = loss
                    best_params = {"lr":lr,"damping":damp,"weigt_decay":wd,"momentum":momentum}
                    best_result = result
            except:
                continue
            
            
        
        
    print(f"Best params: {best_params}")
    print(f"Best loss: {best_loss}")
    
    
    if args.two_level:    
        np.save(npy_dir+"_"+args.R0[:4]+'_params.npy',best_params)
        np.save(npy_dir+"_"+args.R0[:4]+'.npy', best_result)
    else:
        np.save(npy_dir+'_params.npy',best_params)
        np.save(npy_dir+'.npy', best_result)
        
    
    time_elapsed = time.time()-t
    
    #np.save(npy_dir+"_"+args.R0[:4]+'.npy', residuals)

    
    print(f"\tProcessing time: {time_elapsed // 60} minutes {time_elapsed % 60} seconds")
    

"""
if __name__ == '__main__':
    optims = ["NG_BD","NG_BD_heuristic","kfac_standard","kfac"]
    for optim in optims:
        auto_args.optim = optim
        tensorboard_name = optim
        
        if optim in ["kfac","kpsvd","kfac_cor","lzs","kpsvd_d"]:
            two_levels = [0]
            dds = [0]
            lists = [two_levels,dds]
            for el in itertools.product(*lists):
                if el==(0,1):
                    continue
                if el==(1,1) and auto_args.R0=="span":
                    continue
                two,dd = el
                auto_args.two_level = two
                auto_args.dd = dd
                if auto_args.two_level and not auto_args.dd:
                    tensorboard_name+="_2L"
                if auto_args.two_level and auto_args.dd:
                    tensorboard_name+= '_2L_dd'
                
                print(f"Processing for optimizer: {optim} two_level:{two} dd:{dd}")
                print(50*"===")
                auto_args.tensorboard_name = tensorboard_name
                #print("Test:",auto_args.tensorboard_name)
                grid_search(auto_args)
                tensorboard_name = optim
        
                
        else:      
            print(f"Processing for optimizer: {optim}")
            print(50*"===")
            auto_args.tensorboard_name =  tensorboard_name
            grid_search(auto_args)
            
       
"""
if __name__ == '__main__':
    optims = ["NG"]
    for optim in optims:
        cnn_args.optim = optim
        tensorboard_name = optim
        
        if optim in ["kfac","kpsvd","kfac_cor","lzs","kpsvd_d"]:
            two_levels = [0]
            dds = [0]
            lists = [two_levels,dds]
            for el in itertools.product(*lists):
                if el==(0,1):
                    continue
                if el==(1,1) and cnn_args.R0=="span":
                    continue
                two,dd = el
                cnn_args.two_level = two
                cnn_args.dd = dd
                if cnn_args.two_level and not cnn_args.dd:
                    tensorboard_name+="_2L"
                if cnn_args.two_level and cnn_args.dd:
                     tensorboard_name+= '_dd'
                print(f"Processing for optimizer: {optim} two_level:{two} dd:{dd}")
                print(50*"===")
                cnn_args.tensorboard_name = tensorboard_name
                #print("Test:",cnn_args.tensorboard_name)
                grid_search(cnn_args)
                tensorboard_name = optim
        
                
        else:      
            print(f"Processing for optimizer: {optim}")
            print(50*"===")
            cnn_args.tensorboard_name =  tensorboard_name
            grid_search(cnn_args)

    
