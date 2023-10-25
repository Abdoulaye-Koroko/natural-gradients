import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import numpy as np




class StandardDampingKFAC(Optimizer):
    
    """
    Implement KFAC with standard damping technique. 
    Implemented for a purpose of studying the impact of regularization technique on KFAC optimizer
    Only Works with toy neural networks
    
    """
    
    
    def __init__(self,net,damping=1e-3):
        
        """
         net: nn.Module
            The model to train
        damping: float
            The regularization parameter
        """

        self.damping = damping
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self.update_stats = False
        self.device = next(net.parameters()).device
        self.T_cov = T_cov
        self._iteration_counter = 0
        self.T_cov = 1
        
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
        
        super(StandardDampingKFAC, self).__init__(self.params, {})
        
    
    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training and self.update_stats:
            self.state[mod]['x'] = i[0]


    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training and self.update_stats :
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)
            

    def compute_covs(self,group,state):
        """Compute the curvature matrix"""
        x = self.state[group['mod']]['x'].detach().clone()
        gy = self.state[group['mod']]['gy'].detach().clone()
        mod = group['mod']
        
        self.batch_size = x.shape[0]
        
        if group['layer_type'] == 'Conv2d':
            A = F.unfold(x, mod.kernel_size, padding=mod.padding,
                         stride=mod.stride)
            B = gy.contiguous().reshape(A.shape[0], -1, A.shape[-1])
            shape = [A.shape[0]]+list(mod.weight.grad.shape)
            dw = torch.einsum('ijk,ilk->ijl', B, A).contiguous().view(*shape)
            if len(group['params']) == 2: # Bias is not None
                db = torch.sum(B, dim=2)
                dw = torch.cat([dw.contiguous().view(self.batch_size,-1),db.contiguous().view(self.batch_size,-1)],dim=1)
            else: #Bias is None
                dw = dw.contiguous().view(self.batch_size,-1)
        elif group["layer_type"]=="Linear":
        
            dw = torch.bmm(gy.contiguous().view(gy.shape[0],-1,1),x.contiguous().view(x.shape[0],1,-1))
            if len(group['params']) == 2: # Bias is not None
                dw = torch.cat([dw.contiguous().view(self.batch_size,-1),gy.contiguous().view(self.batch_size,-1)],dim=1)
            else: # bias is None
                dw = dw.contiguous().view(self.batch_size,-1) 

        dw = dw.contiguous().view(-1,self.batch_size)

        #Computation of xxt
        if group['layer_type'] == 'Conv2d':
            
            x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
          
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
        
            
        xxt = torch.mm(x, x.t()) / float(x.shape[1])
        
        ggt = torch.mm(gy, gy.t()) / float(gy.shape[1])
       
        Fisher = torch.kron(xxt,ggt) + self.damping*torch.diag(torch.ones(xxt.shape[0]*ggt.shape[0],device=self.device))
        
        state["F"] = Fisher
    
    
    
    
    def step(self,update_params):
    
        for iter,group in enumerate(self.param_groups):
        
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None

            state = self.state[weight]

            if self.update_stats and self._iteration_counter % self.T_cov == 0:   

                self.compute_covs(group, state)

            if update_params:
                F_inv = torch.inverse(state["F"])
                
                g = weight.grad.data
                s = g.shape
                if group['layer_type'] == 'Conv2d':
                    g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
                
                if bias is not None:
                    gb = bias.grad.data
                    g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)

                s = g.shape
                g_precond = F_inv@g.view(-1,1)
                g_precond = g_precond.contiguous().view(*s) 

                if bias is not None:
                    gb = g_precond[:, -1].contiguous().view(*bias.shape)
                    g = g_precond[:, :-1]
                else:
                    gb = None
                g = g.contiguous().view(*weight.shape)

                weight.grad.data = g

                if bias is not None:

                    bias.grad.data = gb

                del state["F"]
        
        if self.update_stats:
            
            self._iteration_counter+=1
                
                    
                    
