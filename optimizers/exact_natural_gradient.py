import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import numpy as np




class ExactNG(Optimizer):
    """
    Implement exact natural gradient method using Shermann Morisson Woodbury formulas. 
    It concerns both the full Fisher matrix and the block-diagonal Fisher matrix.
    
    """
    
    def __init__(self,net,method="NG",damping=1e-3,damping_method='standard',batch_size=64,constraint_norm=False,clipping=1e-2):
        
        """
        Parameters
        -----------
        net: nn.Module
            The model to train
        method: str
            NG for full Fisher matrix and NG_BD for block-diagonal Fisher matrix
        damping: float
            The regularization parameter
        damping_method: str
            The technique used to regularize the curvature. It can be either standard or heuristic
        batch_size: int
            Size of batch for computing the curvature matrix
        constraint_norm: bool
            Wether to scale the gradient with Kl-clipping or not
        clipping: float
            Parameter to scale the gradient with kl-clipping
        """
        self.method = method
        self.damping = damping
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self.update_stats = False
        self.device = next(net.parameters()).device
        self.T_cov = 1
        self._iteration_counter = 0
        self.batch_size = batch_size
        self.constraint_norm = constraint_norm
        self.clipping = clipping
        self.damping_method = damping_method
        
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        
        self.num_layer = len(self.params)
        
        super(ExactNG, self).__init__(self.params, {})
        
        
    
    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training and self.update_stats:
            self.state[mod]['x'] = i[0]


    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training and self.update_stats :
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)
            
    
    def collect_stats(self):
        """Collect the statistics needed for computing the curvature matrix."""
        
        for iter,group in enumerate(self.param_groups):
            
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            
            state = self.state[weight]
            
            x = self.state[group['mod']]['x'].detach().clone()
            gy = self.state[group['mod']]['gy'].detach().clone()
            mod = group['mod']
            
            index = np.random.randint(low=0, high=x.shape[0], size=self.batch_size)
            
            x = x[index]
            gy = gy[index]
            
            self.batch_size = x.shape[0]
            
            del self.state[group['mod']]['x']
            del self.state[group['mod']]['gy']
            
            if group['layer_type'] == 'Conv2d':
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,stride=mod.stride)
                x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
                if bias is not None:
                    ones = torch.ones_like(x[:1])
                    x = torch.cat([x, ones], dim=0)

                gy = gy.data.permute(1, 0, 2, 3)
                T = gy.shape[2] * gy.shape[3]
                gy = gy.contiguous().view(gy.shape[0], -1)
            
            elif group['layer_type']=='Linear':
                x = x.data.t()
                gy = gy.data.t()
                T=1

                if bias is not None:
                    ones = torch.ones_like(x[:1])
                    x = torch.cat([x, ones], dim=0)
                    
            state["x"] = x
            state["gy"] = gy
            state["num_locations"] = T
            
                

        
    
    def GTu_MLP(self,weight, bias, state):
        """compute Diag(G_i^TW_iA_{i-1}) for a MLP layer"""
    
        x = state["x"]
        gy = state["gy"]
        
        g = weight.grad.data
        
        
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)
        
        return torch.diagonal(torch.mm(torch.mm(gy.t(), g), x)).view(-1,1)
    
    
    def GTu_CNN(self,weight, bias, state):
        """compute Diag(G_i^TW_iA_{i-1}) for a CNN layer"""
        
        x = state["x"]
        gy = state["gy"]
        T = state["num_locations"] 
        
        g = weight.grad.data
        s = g.shape
        g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)
            
        D = torch.diagonal(torch.mm(torch.mm(gy.t(),g),x))
        
        
        
        r = [torch.sum(D[b*T:(b+1)*T]) for b in range(self.batch_size)]
        r = torch.tensor(r,device=self.device)
        
        return r.view(-1,1)
                           
        
        
        
        
    
    
    def GTG_MLP(self, state):
        """compute G^TG for a MLP layer""" 
        
        x = state["x"]
        gy = state["gy"]
        
        return torch.mul(x.t()@x,gy.t()@gy)
    
    
    def GTG_CNN(self, state):
        """compute G^TG for a CNN layer""" 
        
        
        x = state["x"]
        gy = state["gy"]
        T = state["num_locations"] 
        
        K = torch.mul(x.t()@x,gy.t()@gy)
        
        GTG = torch.zeros((self.batch_size,self.batch_size)).to(self.device)
        
        for b1 in range(self.batch_size):
            for b2 in range(b1+1):
                GTG[b1,b2] = torch.sum(K[b1*T:(b1+1)*T,b2*T:(b2+1)*T])
        
        
        return GTG+GTG.t()-torch.diag(torch.diagonal(GTG,offset=0))
        
        
    
    
    def Gv_MLP(self, state,v):
        """MAT(Gv_{[i]}) for a MLP layer""" 
        
        x = state["x"]
        gy = state["gy"]
        T = state["num_locations"] 
        
        v = v.view(1,-1)
            
        return torch.mm(gy*v.expand_as(gy),x.t())
    
    
    def Gv_CNN(self, state,v):
        """MAT(Gv_{[i]}) for a CNN layer""" 
        
        
        x = state["x"]
        gy = state["gy"]
        T = state["num_locations"] 
        
        v = v.view(1,-1)
        
        v = v.repeat(T,1)
        v = v.reshape(-1,v.numel())
        
            
        return torch.mm(gy*v.expand_as(gy),x.t())
    
    
    
    def GTu(self):
        """compute Diag(G_i^TW_iA_{i-1}) for the hole network"""
        
        r = torch.zeros((self.batch_size,1)).to(self.device)
        
        for iter,group in enumerate(self.param_groups):
            
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            
            state = self.state[weight] 
                
            if group['layer_type']=='Conv2d':
                r+=self.GTu_CNN(weight, bias, state)
            elif group['layer_type']=='Linear':
                r+=self.GTu_MLP(weight, bias, state)
            
        return r
    
    def GTG(self):
        """compute G^TG for the hole network""" 
            
        r = torch.zeros((self.batch_size,self.batch_size)).to(self.device)
        
        
        for iter,group in enumerate(self.param_groups):
            
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
                
            state = self.state[weight]
            
            if group['layer_type']=='Conv2d':
                r+=self.GTG_CNN(state)
            elif group['layer_type']=='Linear':
                r+=self.GTG_MLP(state)
            
        return r
    
    
    def heuristic_damping(self,state):
        """Implement heuristic damping technique"""
        
        x = state["x"]
        
        gy = state["gy"]
        
        A = torch.mm(x,x.t())/float(x.shape[1])
        
        G = torch.mm(gy,gy.t())/float(gy.shape[1])
        
        _, Sa, _ = torch.svd(A, compute_uv=False)
        
        _, Sg, _ = torch.svd(G, compute_uv=False)
        
        damp = self.damping*Sa[0] + self.damping*Sg[0]+ self.damping**2
        
        
        return damp
    
    def step(self,update_params=False):
        
        if self.update_stats:
            
            self.collect_stats()
            
            self._iteration_counter+=1
            
        if update_params:
            
            fisher_norm = 0. 
            
            if self.method=="NG":
                
                damping = self.damping
                
                w = self.GTu()
                
                GTG = self.GTG()
                
                v = torch.mm(torch.inverse(torch.eye(self.batch_size).to(self.device)+((damping*self.batch_size)**(-1))*GTG),w)
            
            for iter,group in enumerate(self.param_groups):
            
                # Getting parameters
                if len(group['params']) == 2:
                    weight, bias = group['params']
                else:
                    weight = group['params'][0]
                    bias = None

                state = self.state[weight]
                
                if self.method=="NG_BD":
                    
                    if self.damping_method=="standard":
                        
                        damping = self.damping
                        
                    elif self.damping_method =="heuristic":
                        
                        damping = self.heuristic_damping(state)

                    if group['layer_type']=='Conv2d':
                        
                        w = self.GTu_CNN(weight, bias, state)
                        
                    elif group['layer_type']=='Linear':
                        
                        w = self.GTu_MLP(weight, bias, state)
                    
                    if group['layer_type']=='Conv2d':
                        
                        GTG = self.GTG_CNN(state)
                        
                    elif group['layer_type']=='Linear':
                        
                        GTG = self.GTG_MLP(state)
                    
                    v = torch.mm(torch.inverse(torch.eye(self.batch_size).to(self.device)+((damping*self.batch_size)**(-1))*GTG),w)
                
                g = weight.grad.data
                s = g.shape
                
                if group['layer_type'] == 'Conv2d':
                    g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
                
                if bias is not None:
                    gb = bias.grad.data
                    g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)

                if group['layer_type']=='Conv2d':
                    g =(1/damping)*g-self.Gv_CNN(state,v)/(self.batch_size*(damping**2))
                elif group['layer_type']=='Linear':
                    g =(1/damping)*g-self.Gv_MLP(state,v)/(self.batch_size*(damping**2))
                    
                    
                if bias is not None:
                    gb = g[:, -1].contiguous().view(*bias.shape)
                    g = g[:, :-1]
                    
                if self.constraint_norm :
                    fisher_norm += float(torch.abs((weight.grad *g.contiguous().view(*s) ).sum()))
                    if bias is not None:
                        fisher_norm += float(torch.abs((bias.grad * gb).sum()))
                        
                    
                
                weight.grad.data=g.contiguous().view(*s)
                if bias is not None:
                    bias.grad.data = gb  
                    
            if self.constraint_norm :
                scale = (self.clipping / fisher_norm)**0.5
                if scale==0:
                    scale = 1
                for group in self.param_groups:
                    for param in group['params']:
                        param.grad.data *= min(scale,1) 
                
            
            
            
                
            
                
            
        
        
        
            
        
        
         
        
        