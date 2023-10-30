import torch
import argparse
import numpy as np
import time 
import warnings

from utils.data_utils import*
from utils.train_utils import*
from apps.mlp_cnn.models import*
from apps.mlp_cnn.initializations import*
from optimizers.kfac import KFAC
from optimizers.kpsvd import KPSVD
from optimizers.deflation import Deflation
from optimizers.kfac_cor import KFAC_CORRECTED
from optimizers.lanczos import Lanczos
from optimizers.twolevel_kfac import TwolevelKFAC
from optimizers.exact_natural_gradient import ExactNG

warnings.filterwarnings('ignore')

def train(args):
    
        torch.manual_seed(seed)
        
        optim = args.optim
        
        num_epochs = args.num_epochs
        
        device = torch.device("cuda")
        
        data = args.data
        
        batch_size = args.batch_size
        
        result_name = args.result_name
        
        model = args.model
        
        fisher_batch = int(batch_size/4)
        
        lr = args.lr
        
        damping = args.damping
        
        clipping = args.clipping
        
        momentum = args.momentum
        
        weight_decay = args.weight_decay
        
        criterion = nn.CrossEntropyLoss() 
        
        root = model
        
        init_params = args.init_params
        
        if init_params=="sparse":
            
            init_weights = sparse_init_weights
            
        elif init_params=="normal":
            
            init_weights = normal_init_weights
            
        elif init_params=="xavier_normal":
            
            init_weights = xavier_normal_init_weights
            
        elif init_params=="xavier_uniform":
            
            init_weights =  xavier_uniform_init_weights
          
        if data in ["CIFAR10",'CIFAR10_RESTRICTED','SVHN']:
            
                num_classes = 10
                
        elif data=="CIFAR100":
            
            num_classes = 100
            
        if model =="cuda_convnet":
            
            model = cuda_convnet(num_classes).apply(init_weights)
            
        elif model=="resnet18":
            
            model = resnet18(num_classes)
            
        elif model =="resnet34":
            
            model = resnet34(num_classes)
            
        elif model== "resnet50":
            
            model = resnet50(num_classes)
            
        elif model=="resnet110":
            
            model = resnet110()
            
        elif model =="vgg11":
            
            model = vgg11(num_classes)
        
        elif model=="toyCNN":
            model = ToyCNN(num_classes).apply(init_weights)
        
        model = model.to(device)
        
        since = time.time()
        
        data_root = args.data_path
        
        train_samples,val_samples,trainloader,testloader =  read_data_sets(os.path.abspath(data_root + data),batch_size)
        num_iter_per_epoch = len(trainloader)
        total_iter = num_iter_per_epoch*num_epochs
        compute_time = 0.0
        
        if optim=="sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=momentum,nesterov=(momentum!=0),weight_decay=weight_decay)
            preconditioner = None
            
        elif optim=="adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
            preconditioner = None
            
        elif optim=="kfac":
            optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,nesterov=False,weight_decay=weight_decay)
            preconditioner = KFAC(model,damping=damping,pi=False,T_cov=100,T_inv=100,
                 alpha=0.95, constraint_norm=True,clipping=clipping,batch_size=fisher_batch)
            
        elif optim=="kpsvd":
            optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,nesterov=False,weight_decay=weight_decay)
            preconditioner = KPSVD(model,damping=damping,pi=False,T_cov=100,T_inv=100,
                 alpha=0.95, constraint_norm=True,clipping=clipping,batch_size=fisher_batch)
        
        elif optim=="deflation":
            optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,nesterov=False,weight_decay=weight_decay)
            preconditioner = Deflation(model,damping=damping,pi=False,T_cov=100,T_inv=100,
                 alpha=0.95, constraint_norm=True,clipping=clipping,batch_size=fisher_batch)
        
        elif optim=="kfac_cor":
            optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,nesterov=False,weight_decay=weight_decay)
            preconditioner = KFAC_CORRECTED(model,damping=damping,pi=False,T_cov=100,T_inv=100,
                 alpha=0.95, constraint_norm=True,clipping=clipping,batch_size=fisher_batch) 
        
        elif optim=="lanczos":
            optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,nesterov=False,weight_decay=weight_decay)
            preconditioner = Lanczos(model,damping=damping,pi=False,T_cov=100,T_inv=100,
                 alpha=0.95, constraint_norm=True,clipping=clipping,batch_size=fisher_batch)
        
        elif optim=="twolevel_kfac":
            optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,nesterov=False,weight_decay=weight_decay)
            preconditioner = TwolevelKFAC(model,damping=damping,pi=False,T_cov=100,T_inv=100,
                 alpha=0.95,constraint_norm=True,clipping=clipping,batch_size=fisher_batch,coarse_space=args.coarse_space,krylov=args.krylov)
        
        elif optim=="exactNG":
            optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,nesterov=False,weight_decay=weight_decay)
            preconditioner = ExactNG(model,method="NG_BD",damping=damping,damping_method='standard',
                                     batch_size=64,constraint_norm=False,clipping=clipping) 
            
        else:
            message = "Unknown optimizer make sure that you choose the optimizer name between [sgd,adam,kfac,kpsvd,deflation,kfac_cor,lanczos,twolevel_kfac,exactNG]"
            raise ValueError(message) 
           
        
        training_loss = []
        train_itera_loss = []
        train_itera_time = []
        test_loss = []
        train_acc = []
        val_acc = []
        times = []
        iteras = []
        
        early_stopping = EarlyStopping(patience=10, verbose=False)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        
        best_loss = 100
        for epoch in range(num_epochs):
            train_loss = 0
            val_loss = 0
            train_corr = 0
            val_corr = 0
            model.train()
            for iter_tr,batch_tr in enumerate(trainloader):
                itera = epoch * num_iter_per_epoch + iter_tr + 1
                optimizer.zero_grad()
                inputs,labels=batch_tr
                inputs,labels = inputs.to(device),labels.to(device)
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs,labels)
            
                if  preconditioner is not None:
                    if preconditioner._iteration_counter%preconditioner.T_cov==0:
                        preconditioner.update_stats = True
                        index = np.random.randint(low=0, high=batch_size, size=fisher_batch)
                        outputs_fisher = model(inputs[index])
                        with torch.no_grad():
                            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs_fisher.cpu().data, dim=1),
                                                  1).squeeze().to(device)
                        loss_sample = criterion(outputs_fisher,sampled_y)
                        loss_sample.backward(retain_graph=True)
                        preconditioner.step(update_params=False) 
                        optimizer.zero_grad()
                        preconditioner.update_stats = False
                
                
                 
                loss.backward()
                
                if preconditioner is not None:
                    preconditioner.step(update_params=True) # Preconditionnes the gradients with the computed Fisher   
                
                optimizer.step()
                
                
                _, preds = torch.max(outputs, 1)
                train_corr+=torch.sum(preds == labels.data)  
                train_loss+=loss.item()*inputs.size(0)
                train_itera_loss.append(loss.item())
                train_itera_time.append(time.time()-since)
                iteras.append(itera)
                
                
            model.eval()
            for te_iter,batch_te in enumerate(testloader):
                inputs_te,labels_te=batch_te
                inputs_te,labels_te = inputs_te.to(device),labels_te.to(device)
                outputs_te = model(inputs_te)
                loss_te = criterion(outputs_te,labels_te)
                val_loss+=loss_te.item()*inputs_te.size(0)
                _, preds_te = torch.max(outputs_te, 1)
                val_corr+=torch.sum(preds_te == labels_te.data) 
        
            #scheduler.step()        
            
            train_loss/=train_samples
            val_loss/=val_samples
            train_corr=(train_corr/train_samples)*100
            val_corr = (val_corr/val_samples)*100
            
            if train_loss<best_loss:
                best_loss = train_loss
            
            compute_time = time.time()-since
            training_loss.append(train_loss)
            train_acc.append(train_corr)
            test_loss.append(val_loss)
            val_acc.append(val_corr)
            times.append(compute_time)
     
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"\tTrain loss: {train_loss}; Test loss: {val_loss}")
            print(f"\tTrain acc: {train_corr}; Test acc: {val_corr}")
            
            early_stopping(train_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        result = dict(train_loss=torch.tensor(training_loss,device="cpu"),val_loss=torch.tensor(test_loss,device="cpu"),
                      times=times,train_itera_loss=torch.tensor(train_itera_loss,device="cpu"),train_itera_time=train_itera_time,
                      iteras=iteras,train_acc=torch.tensor(train_acc,device="cpu"),val_acc=torch.tensor(val_acc,device="cpu"))
         
        time_elapsed = time.time() - since 
        
        print("\n")
        print("Training Summary:")
        print(f"\tTraining time: {time_elapsed // 60} minutes {time_elapsed % 60} seconds")
        
        return result
    
    



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Function arguments')
    
    parser.add_argument('--model', type=str, default ="cuda_convnet",
                        help='model name')
    parser.add_argument('--optim', type=str, default="kfac",
                        help='optimizer name')
    
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    
    parser.add_argument('--data', type=str, default="CIFAR10",
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
    
    parser.add_argument('--init_params', type=str, default='xavier_uniform',
                        help='Initialization method')
    
    parser.add_argument('--coarse_space', type=str, default='nicolaides',
                        help='Coarse space used in two-level KFAC')
    
    parser.add_argument('--krylov', type=int, default=0,
                        help='Whether to use krylov coarse space or not in two-level KFAC')
    
    
    

    args = parser.parse_args()
    
    results = train(args)
    
    output_folder = "apps/mlp_cnn/results/"
    
    if not os.path.exists(output_folder):
        
        os.makedirs(output_folder)
        
    np.save(output_folder + f'{args.result_name}.npy', results) 
    
    print(f"The results are saved in {output_folder} under the name {args.result_name}.npy")

    
