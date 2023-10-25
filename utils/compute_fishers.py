#This script compute the exact Fisher for a layer and 
# the different approximations associated to it (kpsvd, kfac, etc.)
# Then compute the approximation quality of a given method (kpsvd, kfac, etc)

import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import time

from utils.data_utils import*
from models.models import*
from utils.optim_utils import*


def exact_fisher(a,g):
    """
    Compute the exact Fisher matrix for a layer.
    
    Parameters
    ----------
    a: torch.tensor
        Matrix of activations
    g: torch.tensor
        Matrix of pre-activation derivatives
    
    Returns
    -------
    res: torch.tensor
        Fisher matrix corresponding of the layer whose activations and pre-activation dervatives are a and g.
    """
    
    a,g = a.detach().clone(),g.detach().clone()
    
    m = a.shape[0]
    
    res = 0
    
    for k in range(m):
        
        a_k = a[k]
        
        g_k = g[k]
        
        res+= torch.kron(torch.unsqueeze(a_k,1)@torch.unsqueeze(a_k,1).t(),torch.unsqueeze(g_k,1)@torch.unsqueeze(g_k,1).t())
        
    return (1/m)*res



class FisherQualities(Optimizer):

    def __init__(self, net):
        """
        Compute different types of fisher approximations and compare them the the exact Fisher for a given dense layer.
        """
        self.T_cov = 1
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        self.max_iter = 1000 # for power svd and lanczos algorithms
        self.epsilon = 1e-6 # for power svd and lanczos algorithms
        self.K=2 # for and lanczos algorithm
    
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        super(FisherQualities, self).__init__(self.params, {})
        
            

    def step(self, update_stats=True):
        
        """Performs one step of preconditioning."""
        
        fisher_norm = 0.
        
        for iter ,group in enumerate(self.param_groups):
            
            if iter==4:
                
            # Getting parameters
                if len(group['params']) == 2:
                    
                    weight, bias = group['params']
                    
                else:
                    
                    weight = group['params'][0]
                    
                    bias = None
                    
                state = self.state[weight]
                
            # Update convariances and inverses
                if update_stats:
                    
                    if self._iteration_counter % self.T_cov == 0:
                        
                        self._compute_covs(group, state)
                        
                        u_ex, s_ex, v_ex = torch.svd(state["exact"])
                        
                        u_kfac,s_kfac,v_kfac = torch.svd(torch.kron(state["xxt"],state["ggt"]))
                        
                        u_kp,s_kp,v_kp = torch.svd(torch.kron(state["R"],state["S"]))
                        
                        u_cor,s_cor,v_cor =  torch.svd(torch.kron(state["xxt"],state["ggt"])+torch.kron(state["xxt_cor"],state["ggt_cor"]))
                        
                        u_d,s_d,v_d =  torch.svd(torch.kron(state["R1"],state["S1"])+torch.kron(state["R2"],state["S2"]))
                        
                        u_lzs,s_lzs,v_lzs = torch.svd(torch.kron(state["A"],state["B"])+torch.kron(state["C"],state["D"]))
                        
                        er_kfac = torch.norm(state["exact"]-torch.kron(state["xxt"],state["ggt"]),p='fro')/torch.norm(state["exact"],p="fro")
                        
                        er_kp = torch.norm(state["exact"]-torch.kron(state["R"],state["S"]), p='fro')/torch.norm(state["exact"],p="fro")
                        
                        er_cor = torch.norm(state["exact"]-torch.kron(state["xxt"],state["ggt"])-\
                                            torch.kron(state["xxt_cor"],state["ggt_cor"]), p='fro')/torch.norm(state["exact"],p="fro")
                        
                        er_d = torch.norm(state["exact"]-torch.kron(state["R1"],state["S1"])-\
                                          torch.kron(state["R2"],state["S2"]), p='fro')/torch.norm(state["exact"],p="fro")
                        
                    
                        er_lzs =  torch.norm(state["exact"]-torch.kron(state["A"],state["B"])-\
                                             torch.kron(state["C"],state["D"]), p='fro')/torch.norm(state["exact"],p="fro")
        
                        sp_kfac = torch.norm(s_ex-s_kfac, p=2)/torch.norm(s_ex,p=2)
            
                        sp_kp = torch.norm(s_ex-s_kp, p=2)/torch.norm(s_ex,p=2)
                
                        sp_cor =  torch.norm(s_ex-s_cor, p=2)/torch.norm(s_ex,p=2)
                    
                        sp_lzs =  torch.norm(s_ex-s_lzs, p=2)/torch.norm(s_ex,p=2)
                        
                        sp_d = torch.norm(s_ex-s_d, p=2)/torch.norm(s_ex,p=2)
                        
                        self._iteration_counter+=1
                        
                        return er_kfac,er_kp,er_cor,er_lzs,sp_kfac,sp_cor,sp_kp,sp_lzs,er_d,sp_d
                            
        
       

    def _save_input(self, mod, i):
        
        """Saves input of layer to compute covariance."""
        
        if mod.training:
            
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        
        """Saves grad on output of layer to compute covariance."""
        
        if mod.training:
            
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    
    def _compute_covs(self, group, state):
        
        """Computes the covariances."""
        
        mod = group['mod']
        
        x = self.state[group['mod']]['x']
        
        gy = self.state[group['mod']]['gy']
        
        # Computation of xxt
        x = x.data.t()
        
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
          
    
        gy = gy.data.t()
        
        state['num_locations'] = 1
        
        #KFAC   
        state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        
        state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        
        #KPSVD
        if self._iteration_counter==0:
            
            v = torch.rand(gy.shape[0]*gy.shape[0],device=gy.get_device())
            
        else:
            
            v = vec(state["S"])
            
        state['R'],state['S'] = power_svd_mlp(x.t(),gy.t(),v,None
                  ,None,self.epsilon,self.max_iter,method="kpsvd")
         
        #LANCZOS  
        if self._iteration_counter==0:
            
            q = torch.rand(gy.shape[0]*gy.shape[0],device=gy.get_device())
            
        else:
            
            q = vec(state["B"]).squeeze()
    
        state['A'],state['B'],state["C"],state["D"] = lanczos_mlp(x.t(),gy.t(),q,self.K,self.epsilon)
        
        # KFAC corrected  
        if self._iteration_counter==0:
            
            v = torch.rand(gy.shape[0]*gy.shape[0],device=gy.get_device())
            
        else:
            
            v = vec(state["ggt_cor"])
        
        state['xxt_cor'],state['ggt_cor'] = power_svd_mlp(x.t(),gy.t(),v,state["xxt"]
                  ,state["ggt"],self.epsilon,self.max_iter,method="kfac_cor")
        
        
        #DEFLATION
        
        if self._iteration_counter==0:
            
            v1 = torch.rand(gy.shape[0]*gy.shape[0],device=gy.get_device())
            
            v2 = torch.rand(gy.shape[0]*gy.shape[0],device=gy.get_device())
            
        else:
            
            v1 = vec(state["S1"])
            
            v2 = vec(state["S2"])
            
        state['R1'],state['S1'] = power_svd_mlp(x.t(),gy.t(),v1,None
                  ,None,self.epsilon,self.max_iter,method="kpsvd")
        
        state['R2'],state['S2'] = power_svd_mlp(x.t(),gy.t(),v2,state["R1"]
                  ,state["S1"],self.epsilon,self.max_iter,method="deflation")
    
            
        #Exact Fisher
        state["exact"] = exact_fisher(x.t(),gy.t())
        
    
    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
            

            
def train(device,num_epochs,pi=True,epsilon=1e-6,max_iter=100,data='CURVES',batch_size=256):
    
    lr = 1e-3
    iterations = []
    erreur_kfac = []
    erreur_kpsvd = []
    erreur_cor =[]
    erreur_lzs = []
    erreur_kfac_sp = []
    erreur_kpsvd_sp = []
    erreur_cor_sp = []
    erreur_lzs_sp = []
    erreur_d = []
    erreur_d_sp = []
    
    n_train,n_val,trainloader,testloader = read_data_sets(os.path.abspath('../../data/' + data),batch_size)
    
    num_iter_per_epoch = len(trainloader)
    
    total_iter = num_epochs*num_iter_per_epoch
    
    if data=="FACES":
        model = autoencoder_faces().to(device)
        criterion = nn.MSELoss()
    elif data=="CURVES":
        model = autoencoder_curves().to(device)
        criterion = nn.BCELoss()
    elif data=="MNIST":
        model = autoencoder_mnist().to(device)
        criterion=nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    precond = FisherQualities(net=model)
    
    '''
    PATH = data+"model.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    results = checkpoint["results"]
    
    
    itera = checkpoint["iter"]
    start_epoch=checkpoint["epoch"]
    
    iterations = results["iterations"]
    erreur_kfac = results["kfac"]
    erreur_kpsvd = results["kpsvd"]
    erreur_cor = results["kfac_cor"]
    erreur_lzs = results["lzs"]
    erreur_kfac_sp = results["kfac_sp"]
    erreur_kpsvd_sp = results["kpsvd_sp"]
    erreur_cor_sp = results["kfac_cor_sp"]
    erreur_lzs_sp = results["lzs_sp"]
    
    erreur_d = results["kpsvd_d"]
    erreur_d_sp = results[""kpsvd_d_sp""]
    iteras_d = results["iteras_d"]
    betas_d = results["betas_d"]
    
    iteras_lzs = results["iteras_lzs"]
    iteras_cor = results["iteras_cor"]
    iteras_kp  = results["iteras_kp"]
    betas_lzs = results["betas_lzs"]
    betas_cor = results["betas_cor"]
    betas_kp = results["betas_kp"]
    
    '''
    itera = 0
    
    start_epoch = 0
    
    for epoch in range(start_epoch,num_epochs):
        
        for iter,batch in enumerate(trainloader):
            
            #itera = epoch * num_iter_per_epoch + iter + 1
            itera+=1
            
            optimizer.zero_grad()
            
            inputs,labels = batch
            
            inputs = inputs.view(inputs.size(0),-1)
            
            inputs,labels = inputs.to(device),labels.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs,inputs)
            
            if data in ["CURVES","MNIST"]:
                
                    sample_y = torch.bernoulli(outputs)
                    
            elif data == "FACES":
                
                    sample_y = torch.normal(mean=outputs,std=torch.ones(outputs.size(1),device=outputs.get_device()))
                
            loss_sample = criterion(outputs,sample_y.detach())
            
            loss_sample.backward(retain_graph=True)
            
        
            er_kfac,er_kp,er_cor,er_lzs,sp_kfac,sp_cor,sp_kp,sp_lzs,er_d,sp_d = precond.step()
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step() 
            
            print(f"Iter: {itera}/{total_iter}")
            
            print(f"\tLoss: {loss}")
                                                   
            print(f"\tError kfac: {er_kfac}")
            
            print(f"\tError kpsvd: {er_kp}")
            
            print(f"\tError kfac_cor: {er_cor}")
            
            print(f"\tError lzs: {er_lzs}") 
            
            print(f"\tError kpsvd_d: {er_d}")
                                                   
            
            print(f"\tError spectre kfac: {sp_kfac}")
            
            print(f"\tError spectre kpsvd: {sp_kp}")
            
            print(f"\tError spectre kfac_cor: {sp_cor}")
            
            print(f"\tError spectre lzs: {sp_lzs}")
            
            print(f"\tError spectre kpsvd_d: {sp_d}")
                                                   
            iterations.append(itera)
            erreur_kfac.append(er_kfac.cpu())
            erreur_kpsvd.append(er_kp.cpu())   
            erreur_cor.append(er_cor.cpu())
            erreur_lzs.append(er_lzs.cpu())
            erreur_d.append(er_d.cpu())
            
            erreur_kfac_sp.append(sp_kfac.cpu())
            erreur_kpsvd_sp.append(sp_kp.cpu())
            erreur_cor_sp.append(sp_cor.cpu())
            erreur_lzs_sp.append(sp_lzs.cpu())
            erreur_d_sp.append(sp_d.cpu())
            
            
            results = {"iterations":iterations,"kfac":erreur_kfac,"kfac_sp":erreur_kfac_sp,
                       "kpsvd":erreur_kpsvd,"kpsvd_sp":erreur_kpsvd_sp,
                        "kfac_cor":erreur_cor,"lzs":erreur_lzs,"kfac_cor_sp":erreur_cor_sp,
                       "lzs_sp":erreur_lzs_sp,"kpsvd_d":erreur_d,"kpsvd_d_sp":erreur_d_sp}
        
            PATH = data+"model.pt"
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'iter': itera,
            'results':results 
            }, PATH)
            
                
                                                   
                                            
                                                   
                                                   
    
    results = {"iterations":iterations,"kfac":erreur_kfac,"kfac_sp":erreur_kfac_sp,
                       "kpsvd":erreur_kpsvd,"kpsvd_sp":erreur_kpsvd_sp,
                        "kfac_cor":erreur_cor,"lzs":erreur_lzs,"kfac_cor_sp":erreur_cor_sp,
                       "lzs_sp":erreur_lzs_sp,"kpsvd_d":erreur_d,"kpsvd_d_sp":erreur_d_sp}   
    
    np.save("fishers_"+data+".npy",results)
    
    return results
    
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
if __name__=="__main__": 
    
    t=time.time()
    datasets = ["CURVES"]
    device = torch.device("cuda")
    batch_size = 256
    for data in datasets:
        print(f"Processing for {data}:")
        print("\n")
        if data=="CURVES":
            num_epochs=40
        elif data=="MNIST":
            num_epochs = 26
            batch_size = 512
        elif data=="FACES":
            batch_size = 1024
            num_epochs=32
        results =  train(device,num_epochs,pi=True,epsilon=1e-6,max_iter=100,data=data,batch_size=batch_size)
    print(f"Processing time : {(time.time()-t)/60} minutes")

    
        
                
                



