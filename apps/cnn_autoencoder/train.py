import torch
import argparse
import numpy as np
import time 

from utils.data_utils import*
from utils.train_utils import*
from apps.cnn_autoencoder.models import*
from optimizers.kfac import KFAC

seed = 0

def train(args):
    
        torch.manual_seed(seed)
        
        optim = args.optim
        
        num_epochs = args.num_epochs
        
        device = torch.device("cuda")
        
        data = args.data
        
        batch_size = args.batch_size
    
        tensorboard_name = args.tensorboard_name
        
        model = args.model
        
        fisher_batch = int(batch_size/4)
        
        lr = args.lr
        
        damping = args.damping
        
        clipping = args.clipping
        
        momentum = args.momentum
        
        weight_decay = args.weight_decay
        
        criterion = nn.CrossEntropyLoss() 
        
        root = model
        
        if model =="cnn_autoencoder_mnist":
            
            model = cnn_autoencoder_mnist()
            
            criterion = nn.BCELoss()
        
        elif model == "cnn_autoencoder_cifar":
            
            model = cnn_autoencoder_cifar()
            
            criterion = nn.MSELoss() 
        
        elif model == "cnn_autoencoder_svhn":
            
            model = cnn_autoencoder_svhn()
            
            criterion = nn.MSELoss() 
            
        model = model.to(device)
        
        since = time.time()
        
        train_samples,val_samples,trainloader,testloader =  read_data_sets(os.path.abspath('../../data/' + data),batch_size)
        
        num_iter_per_epoch = len(trainloader)
        
        total_iter = num_iter_per_epoch*num_epochs
        
        frequency = 100
        
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
        else:
            message = "Unknown optimizer make sure that you choose here the optimizer name between [sgd,adam,kfac]"
            raise ValueError(message) 
            
        
        training_loss = []
        
        train_itera_loss = []
        
        train_itera_time = []
        
        test_loss = []
        
        times = []
        
        iteras = []
        
        early_stopping = EarlyStopping(patience=10, verbose=False)
        
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        
        best_loss = 100
        
        for epoch in range(num_epochs):
            
            train_loss = 0
            
            val_loss = 0
            
            model.train()
            
            for iter_tr,batch_tr in enumerate(trainloader):
                
                itera = epoch * num_iter_per_epoch + iter_tr + 1
                
                optimizer.zero_grad()
                
                inputs,labels=batch_tr
                
               #batch_size = inputs.size(0)
                inputs,labels = inputs.to(device),labels.to(device)
                
                with torch.set_grad_enabled(True):
                    
                    outputs = model(inputs)
                        
                    loss = criterion(outputs,inputs)
                        
                if  preconditioner is not None:

                    if preconditioner._iteration_counter%preconditioner.T_cov==0:

                        preconditioner.update_stats = True

                        index = np.random.randint(low=0, high=batch_size, size=fisher_batch)

                        outputs_fisher = model(inputs[index])

                        with torch.no_grad():

                            if data in ["MNIST"]:

                                sampled_y = torch.bernoulli(outputs_fisher)

                            elif data in ["CIFAR10","SVHN"]:

                                sampled_y = torch.normal(mean=outputs_fisher)


                        loss_sample = criterion(outputs_fisher,sampled_y.detach())

                        loss_sample.backward(retain_graph=True)

                        preconditioner.step(update_params=False) 

                        optimizer.zero_grad()

                        preconditioner.update_stats = False
                
                
                 
                loss.backward()
                
                if preconditioner is not None:
                    
                    preconditioner.step(update_params=True) # Preconditionnes the gradients with the computed Fisher   
                
                optimizer.step()
                
                train_loss+=loss.item()*inputs.size(0)
                
                train_itera_loss.append(loss.item())
                
                train_itera_time.append(time.time()-since)
                
                iteras.append(itera)
                
                
            model.eval()
            
            for te_iter,batch_te in enumerate(testloader):
                
                inputs_te,labels_te=batch_te
                
                inputs_te,labels_te = inputs_te.to(device),labels_te.to(device)
                
                outputs_te = model(inputs_te)
                    
                loss_te = criterion(outputs_te,inputs_te)
                    
                val_loss+=loss_te.item()*inputs_te.size(0)
                
            #scheduler.step()        
            
            train_loss/=train_samples
            
            val_loss/=val_samples
            
            if train_loss<best_loss:
                
                best_loss = train_loss
            
            compute_time = time.time()-since
            
            training_loss.append(train_loss)
            
            test_loss.append(val_loss)
            
            times.append(compute_time)
     
            print(f"Epoch {epoch}/{num_epochs}:")
        
            print(f"\tTrain loss: {train_loss}; Test loss: {val_loss}")
            
            early_stopping(train_loss, model)
        
            if early_stopping.early_stop:
                
                print("Early stopping")
                
                break
                    
                     
        result = dict(train_loss=torch.tensor(training_loss,device="cpu"),val_loss=torch.tensor(test_loss,device="cpu"),times=times,
                      train_itera_loss=torch.tensor(train_itera_loss,device="cpu"),train_itera_time=train_itera_time,
                      iteras=iteras)
        
        time_elapsed = time.time() - since 

    
        print("\n")
        
        print("Training Summary:")
        
        print(f"\tTraining time: {time_elapsed // 60} minutes {time_elapsed % 60} seconds")
        
        
        return best_loss,result 
    



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Function arguments')
    
    parser.add_argument('--model', type=str, default ="cnn_autoencoder_mnist",
                        help='model name')
    parser.add_argument('--optim', type=str, default="kfac",
                        help='optimizer name')
    
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    
    parser.add_argument('--data', type=str, default="MNIST",
                        help='data name')
    
    parser.add_argument('--tensorboard_name', type=str, default="kfac",
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
    
    results = train(args)

    
