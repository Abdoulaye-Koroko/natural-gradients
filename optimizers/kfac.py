import torch
from optimizers.base_optimzer import BaseOptimizer



class KFAC(BaseOptimizer):

    def __init__(self,net,damping=1e-3,pi=False,T_cov=100,T_inv=100,
                 alpha=0.95, constraint_norm=False,kl=1e-2,batch_size=64):
        
        """ 
        Inspired from https://github.com/Thrandis/EKFAC-pytorch. 
        I extended the original KFAC to handle Transposed convolutional layers.
        It works for Linear, Conv2d layers and ConvTranspose2d, and silently skip other layers.
        
         Parameters
        -----------
        net: nn.Module
            The model to train
        damping: int
            The regularization parameter
        pi: bool
            Whether to use to scale the regularization parmeter with traces of Kronecker 
            factors or not
        T_cov: int
            The frequence at which the curvature matrix is updated
        T_inv: int
            The frequence at which the inverse of the curvature matrix is computed
        alpha: float
            Parameter for exponentially moving average
        constraint_norm: bool
            Wether to scale the gradient with Kl-clipping or not
        kl: float
            Parameter to scale the gradient with kl-clipping
        batch_size: int
            Size of batch for computing the curvature matrix
        """
        
        super(KFAC, self).__init__(self.params, {})
        
        self.damping = damping
        self.pi = pi
        self.T_cov = T_cov
        self.T_inv = T_inv
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.kl = kl
        self.batch_size = batch_size
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d','ConvTranspose2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.param_groups.append(d)
        

    def step(self,update_params=True):
        
        """Performs one step of preconditioning."""
        fisher_norm = 0.

        for group in self.param_groups:
            
            # Getting parameters
            if len(group['params']) == 2:
                
                weight, bias = group['params']
                
            else:
                
                weight = group['params'][0]
                
                bias = None
                
            state = self.state[weight]
            
            # Update convariances and inverses
            if self.update_stats:
                
                if self._iteration_counter % self.T_cov == 0:
                    
                    self._compute_covs(group, state)
                    
                if self._iteration_counter % self.T_inv == 0:
                    
                    ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'],
                                                state['num_locations'])
                    state['ixxt'] = ixxt
                    
                    state['iggt'] = iggt
                
            if update_params:
                
                # Preconditionning
                gw, gb = self._precond(weight, bias, group, state)
                
                # Updating gradients
                if self.constraint_norm:
                    
                    fisher_norm += torch.abs((weight.grad * gw).sum())
                    
                weight.grad.data = gw
                
                
                if bias is not None:
                    
                    if self.constraint_norm:
                        
                        fisher_norm += torch.abs((bias.grad * gb).sum())
                        
                    bias.grad.data = gb
                    
            # Cleaning
            if 'x' in self.state[group['mod']]:
                
                del self.state[group['mod']]['x']
                
            if 'gy' in self.state[group['mod']]:
                
                del self.state[group['mod']]['gy']
                
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            
            fisher_norm += 1e-8
            
            scale = min((self.clipping/fisher_norm) ** 0.5,1.0)
            
            for group in self.param_groups:
                
                for param in group['params']:
                    
                    param.grad.data *= scale
         
        if self.update_stats:
            
            self._iteration_counter += 1

    
    
    def _save_input(self, mod, i):
        
        """Saves input of layer to compute covariance."""
        
        if mod.training and self.update_stats:
            
            self.state[mod]['x'] = i[0]


    def _save_grad_output(self, mod, grad_input, grad_output):
        
        """Saves grad on output of layer to compute covariance."""
        
        if mod.training and self.update_stats:
            
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)
            

    def _precond(self, weight, bias, group, state):
        
        """Applies preconditioning."""
        
        ixxt = state['ixxt']
        
        iggt = state['iggt']
        
        g = weight.grad.data
        
        s = g.shape
        
        #print(f"weight shape: {g.shape}")
            
        #print(f"bias shape: {bias.grad.shape}")
        
        if group['layer_type'] == 'Conv2d':
            
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        
        elif group['layer_type'] == 'ConvTranspose2d':
            
            #g = g.transpose(0,1)
            
            g = torch.rot90(g,k=2,dims=[2,3])
            
            g = g.contiguous().view(s[1], s[0]*s[2]*s[3])
            
        if bias is not None:
            
            gb = bias.grad.data
            
            #print(f"weight shape: {g.shape}")
            
            #print(f"bias shape: {gb.shape}")
            
            #print(40*"+++")
            
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
            
        g = torch.mm(torch.mm(iggt, g), ixxt)
        
        if group['layer_type'] in ['Conv2d','ConvTranspose2d']:
            
            g /= state['num_locations']
            
        if bias is not None:
            
            gb = g[:, -1].contiguous().view(*bias.shape)
            
            g = g[:, :-1]
            
        else:
            
            gb = None
            
        g = g.contiguous().view(*s)
        
        if group["layer_type"] == 'ConvTranspose2d':
            
            g = torch.rot90(g,k=-2,dims=[2,3])
        
        return g, gb


    def unfold_ConvTranspose2d(self,inputs,kernel_size,padding,stride):
        
        """Compute the unfolding map for ConvTranspose2d layers"""
        
        def stride_fractionally(x, stride):
    
            """Put stride rows and columns of zeros between rows and columns of x respectively"""
        
            stride_w, stride_h = stride

            w = x.new_zeros(stride_w, stride_h)

            w[0, 0] = 1

            res = F.conv_transpose2d(x, w.expand(x.size(1), 1, stride_w, stride_h), stride=stride, groups=x.size(1))

            s_w,s_h = max(stride_w-1,1),max(stride_h-1,1)

            return res[:,:,:-s_w,:-s_h]
    
        if min(stride)>1:
        
            inputs = stride_fractionally(inputs, stride=stride)
         
        pad_w = max(kernel_size[0]-padding[0]-1,0)
        
        pad_h = max(kernel_size[1]-padding[1]-1,0)
        
        #pad = kernel_size-padding-1
     
        input_unfolded = F.unfold(inputs,kernel_size=kernel_size,padding=(pad_w,pad_h),stride=1)
        
        return input_unfolded

    
    
    
    def _compute_covs(self, group, state):
        
        """Computes the covariances."""
        
        mod = group['mod']
        
        x = self.state[group['mod']]['x']
        
        gy = self.state[group['mod']]['gy']
        
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            
            
            x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            
        
        elif group['layer_type'] == 'ConvTranspose2d':
            
            x = self.unfold_ConvTranspose2d(x,mod.kernel_size,padding=mod.padding,stride=mod.stride)
            
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        
        else: # layer_type == Linear
            
            x = x.data.t()
            
        if mod.bias is not None:
            
            ones = torch.ones_like(x[:1])
            
            x = torch.cat([x, ones], dim=0)
            
        if self._iteration_counter == 0:
            
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
            
        else:
            
            rho = min(1-1/self._iteration_counter,self.alpha)
            
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=rho,
                                alpha=(1-rho)/ float(x.shape[1]))
        
        # Computation of ggt
        if group['layer_type'] in ['Conv2d','ConvTranspose2d']:
            
            gy = gy.data.permute(1, 0, 2, 3)
            
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            
            gy = gy.contiguous().view(gy.shape[0], -1)
            
        else: # layer_type == Linear
            
            gy = gy.data.t()
            
            state['num_locations'] = 1
            
        if self._iteration_counter == 0:
            
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
            
        else:
            
            rho = min(1-1/self._iteration_counter,self.alpha)
            
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=rho,
                                alpha=(1-rho) / float(gy.shape[1]))

    
    
    
    def _inv_covs(self, xxt, ggt, num_locations):
            
        """Inverses the covariances."""
        # Computes pi
        
        pi = 1.0
        
        damping = self.damping/num_locations
        
        if self.pi:
            
            tx = torch.trace(xxt) * ggt.shape[0]
            
            tg = torch.trace(ggt) * xxt.shape[0] + 1e-8
            
            pi = (torch.abs(tx / tg))**0.5
            
            if any([pi==np.inf, pi==0, torch.isnan(pi)]):
                
                pi = 1.0
        # Regularizes and inverse
        
        #print(f"pi : {pi}")
        x_add = pi*(damping)**0.5
        
        g_add = (1.0/pi)*(damping)**0.5
            
        
        #print(f"x_add: {x_add}, g_add: {g_add}") 
        
        diag_xxt = xxt.new(xxt.shape[0]).fill_(x_add)
        
        diag_ggt = ggt.new(ggt.shape[0]).fill_(g_add)
        
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        
        return ixxt, iggt