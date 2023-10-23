import torch
import argparse
import numpy as np
import time 

from utils.data_utils import*
from utils.train_utils import*
from models.models import*
from models.initializations import*
from optimizers.kfac import KFAC
from optimizers.kpsvd import KPSVD

def train(args):

        torch.manual_seed(0)
        optim = args.optim
        num_epochs = args.num_epochs
        device = torch.device("cuda")
        data = args.data
        batch_size = args.batch_size
        tensorboard_name = args.tensorboard_name
        fisher_batch = int(batch_size/4)
        lr = args.lr
        damping = args.damping
        clipping = args.clipping
        momentum = args.momentum
        weight_decay = args.weight_decay
        root = data
        init_params = args.init_params
        
        if init_params=="sparse":
            init_weights = sparse_init_weights
        elif init_params=="normal":
            init_weights = normal_init_weights
        elif init_params=="xavier_normal":
            init_weights = xavier_normal_init_weights
        elif init_params=="xavier_uniform":
            init_weights =  xavier_uniform_init_weights  
        
        if data=="FACES":
            net = autoencoder_faces().apply(init_weights)
            model = net.to(device)
            criterion = nn.MSELoss() 
        elif data=="CURVES":
            net = autoencoder_curves().apply(init_weights)
            model = net.to(device)
            criterion = nn.BCELoss()
        elif data=="MNIST":
            net = autoencoder_mnist().apply(init_weights)
            model = net.to(device)
            criterion = nn.BCELoss()
        
        elif data=="F_MNIST":
            net = DeepLinear().apply(init_weights)
            model = net.to(device)
            criterion = nn.CrossEntropyLoss()  
            
        elif data=="D_MNIST":
            input_dim = 784
            net = MLP(input_dim,num_classes=10).apply(init_weights)
            model = net.to(device)
            criterion = nn.CrossEntropyLoss()
            
        elif data=="MNIST_RESTRICTED":
            net =  ToyNet(10).apply(init_weights)
            model = net.to(device)
            criterion = nn.CrossEntropyLoss()
        
        since = time.time()
        data_root = "../../data/"
        train_samples,val_samples ,trainloader,testloader = read_data_sets(os.path.abspath(data_root + data),batch_size)
        
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
        
        training_loss = []
        train_itera_loss = []
        train_itera_time = []
        test_loss = []
        train_acc = []
        val_acc = []
        times = []
        iteras = []
        early_stopping = EarlyStopping(patience=10, verbose=False)
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
                inputs = inputs.view(inputs.size(0),-1).to(device)
                inputs,labels = inputs.to(device),labels.to(device)
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    if data in ["F_MNIST","CIFAR10","D_MNIST","MNIST_RESTRICTED"]:
                        loss = criterion(outputs,labels)
                    else: 
                        loss = criterion(outputs,inputs)
                
                if preconditioner is not None:
                    if preconditioner._iteration_counter%preconditioner.T_cov==0:
                        preconditioner.update_stats = True
                        index = np.random.randint(low=0, high=batch_size, size=fisher_batch)
                        outputs_fisher = model(inputs[index])
                        if data in ["CURVES","MNIST"]:
                            with torch.no_grad():
                                sample_y = torch.bernoulli(outputs_fisher)
                        elif data in ["FACES"]:
                            with torch.no_grad():
                                sample_y = torch.normal(mean=outputs_fisher)
                        elif data in ["F_MNIST","CIFAR10","D_MNIST","MNIST_RESTRICTED"]:
                            with torch.no_grad():
                                sample_y = torch.multinomial(torch.nn.functional.softmax(outputs_fisher.cpu().data, dim=1),
                                                  1).squeeze().to(device)
                        loss_sample = criterion(outputs_fisher,sample_y.detach())
                        loss_sample.backward(retain_graph=True)
                        preconditioner.step(update_params=False) #Compute the Fisher with sampled examples
                        optimizer.zero_grad()
                        preconditioner.update_stats = False
                loss.backward()
                
                if preconditioner is not None:
                    
                    preconditioner.step(update_params=True) # Preconditionnes the gradients with the computed Fisher 
                optimizer.step()
                
                if data in ["F_MNIST","CIFAR10","D_MNIST","MNIST_RESTRICTED"]:
                    _, preds = torch.max(outputs, 1)
                    train_corr+=torch.sum(preds == labels.data)  
                
                train_loss+=float(loss)*inputs.size(0)
                train_itera_loss.append(float(loss))
                train_itera_time.append(time.time()-since)
                iteras.append(itera)
                
                del inputs,labels
                
            model.eval()
            for te_iter,batch_te in enumerate(testloader):
                inputs_te,labels_te=batch_te
                inputs_te = inputs_te.view(inputs_te.size(0),-1).to(device)
                inputs_te,labels_te = inputs_te.to(device),labels_te.to(device)
                outputs_te = model(inputs_te)
                if data in ["F_MNIST","CIFAR10","D_MNIST","MNIST_RESTRICTED"]:
                    loss_te = criterion(outputs_te,labels_te)
                else:
                    loss_te = criterion(outputs_te,inputs_te)
                
                
                val_loss+=float(loss_te)*inputs_te.size(0)
                
                if data in ["F_MNIST","CIFAR10","D_MNIST","MNIST_RESTRICTED"]:
                    _, preds_te = torch.max(outputs_te, 1)
                    val_corr+=torch.sum(preds_te == labels_te.data) 
            
                del inputs_te,labels_te
                
                    
        
            train_loss/=train_samples
            val_loss/=val_samples
            
            if train_loss<best_loss:
                best_loss = train_loss
            
            if data in ["F_MNIST","CIFAR10","D_MNIST","MNIST_RESTRICTED"]:
                train_corr=(train_corr/train_samples)*100
                val_corr = (val_corr/val_samples)*100
            
            compute_time = time.time()-since
            
                
            training_loss.append(train_loss)
            test_loss.append(val_loss)
            
            times.append(compute_time)
            
            if data in ["F_MNIST","CIFAR10","D_MNIST","MNIST_RESTRICTED"]:
                train_acc.append(train_corr)
                val_acc.append(val_corr)
     
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"\tTrain loss: {train_loss}; Test loss: {val_loss}")
            
            if data in ["F_MNIST","CIFAR10","D_MNIST","MNIST_RESTRICTED"]:
                print(f"\tTrain acc: {train_corr}; Test acc: {val_corr}")
            
            early_stopping(train_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        if data in ["F_MNIST","CIFAR10","D_MNIST","MNIST_RESTRICTED"]:
            result = dict(train_loss=torch.tensor(training_loss,device="cpu"),val_loss=torch.tensor(test_loss,device="cpu"),times=times,
                      train_itera_loss=torch.tensor(train_itera_loss,device="cpu"),train_itera_time=train_itera_time,
                      iteras=iteras,train_acc=torch.tensor(train_acc,device="cpu"),val_acc=torch.tensor(val_acc,device="cpu"))
        else:   
        
            result = dict(train_loss=torch.tensor(training_loss,device="cpu"),val_loss=torch.tensor(test_loss,device="cpu"),times=times,
                      train_itera_loss=torch.tensor(train_itera_loss,device="cpu"),train_itera_time=train_itera_time,
                      iteras=iteras)
        
        time_elapsed = time.time() - since 
        
        #residuals = dict(residuals=preconditioner.residuals,iterations=preconditioner.iterations)
        
        print("\n")
        print("Training Summary:")
        print(f"\tTraining time: {time_elapsed // 60} minutes {time_elapsed % 60} seconds")
        
        return result
    
    



if __name__=="__main__":
    
    
    parser = argparse.ArgumentParser(description='Function arguments')
    
    parser.add_argument('--optim', type=str, default="kfac",
                        help='optimizer name')
    
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    
    parser.add_argument('--data', type=str, default="CURVES",
                        help='data name')
    
    parser.add_argument('--tensorboard_name', type=str, default="kfac",
                        help='tensorbord_name name')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    
    
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    
    
    parser.add_argument('--damping', type=float, default=1e-4,
                        help='damping')
    
      
    parser.add_argument('--clipping', type=float, default=1e-2,
                        help='kl clipping')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum parameter')
    
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='weight_decay parameter')
    
    parser.add_argument('--init_params', type=str, default='xavier_uniform',
                        help='Weights initialization strategy')
    

    args = parser.parse_args()
    
    results = train(args)
    
