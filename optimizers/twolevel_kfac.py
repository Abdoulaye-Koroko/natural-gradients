import torch
import torch.nn.functional as F
import numpy as np

from optimizers.base_optimizer import BaseOptimizer

class TwolevelKFAC(BaseOptimizer):

    def __init__(self,net,damping=1e-3,pi=False,T_cov=100,T_inv=100,
                 alpha=0.95, constraint_norm=False,clipping=1e-2,batch_size=64,coarse_space="nicolaides",krylov=False):
        
        """ 
        Implement two level KFAC optimizer
        It works for Linear, Conv2d layers and silently skip other layers.
        
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
        clipping: float
            Parameter to scale the gradient with kl-clipping
        batch_size: int
            Size of batch for computing the curvature matrix
        coarse: str
            The coarse space used to cumpute the coarse correction. It can be one of the elements of the following list:
            [nicolaides,spectral,residual,tpselepedis]
        krylov: bool
            Whether to use Krylov subspace to enrich the coarse space or not
        """
        
        super(TwolevelKFAC, self).__init__()
        self.param_groups.pop()
        self.damping = damping
        self.pi = pi
        self.T_cov = T_cov
        self.T_inv = T_inv
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.clipping = clipping
        self.batch_size = batch_size
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self.update_stats = False
        self._iteration_counter = 0
        self.coarse_space = coarse_space
        self.krylov = krylov
        self.flat_grads = None
        self.f_coarse = None
        self.f_coarse_inv = None
        self.errors  = []
        self.device = next(net.parameters()).device
        self.svd = True
        
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
                self.param_groups.append(d)
        
        self.num_layer = len(self.param_groups)
        
        
    def step(self, update_params=True):
        """Performs one step of preconditioning."""

        if self.update_stats:
            
            if self._iteration_counter % self.T_cov == 0:
                
                self.collect_stats()
                    
        if  update_params:
            self.flat_grads = self.gather_list_grad()
        
        for iter,group in enumerate(self.param_groups):
            
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
                    
                    self._compute_covs(group, state,iter)
                
                if self._iteration_counter%self.T_inv==0:
                    
                    if self.svd:

                        eps = 1e-12
                        
                        #print(f"NAN values proprtions in xxt: {torch.isnan(state['xxt']).sum()}")
                        
                        #print(f"NAN values proprtions in ggt: {torch.isnan(state['ggt']).sum()}")

                        L_x, Q_x = torch.linalg.eigh(state["xxt"]+self.damping*torch.diag(torch.ones(state["xxt"].shape[0],
                                                                                                    device = self.device)))

                        L_g,Q_g = torch.linalg.eigh(state["ggt"]+self.damping*torch.diag(torch.ones(state["ggt"].shape[0],
                                                                                                   device = self.device)))

                        L_x.mul_((L_x > eps).float())

                        L_g.mul_((L_g > eps).float())

                        state["L_x"] = L_x

                        state["L_g"] = L_g

                        state["Q_x"] = Q_x

                        state["Q_g"] = Q_g

                    else:

                        ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'],state['num_locations'])

                        state['ixxt'] = ixxt

                        state['iggt'] = iggt
                            
            if update_params:
                
                # Preconditionning
                gw, gb = self._precond(weight, bias, group, state)
                
                # Updating gradients
                weight.grad.data = gw
                
                if bias is not None:
                        
                    bias.grad.data = gb         
   
        if update_params:
        
            fisher_norm = self.precond_two_level() 
            #_ = self.precond_two_level() # for checking residual
                
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            
            scale = (self.clipping / fisher_norm)**0.5
            for group in self.param_groups:
                
                for param in group['params']:
                    
                    param.grad.data *= min(scale,1)
        
        
        if update_params:
            
            self._iteration_counter += 1
            

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if self.svd:

            return self.svd_precond(weight,bias,group,state)
        
        ixxt = state['ixxt']
        iggt = state['iggt']
        
        g = weight.grad.data
        
        s = g.shape
        
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)
        
        
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb
    
    
    def svd_precond(self,weight,bias,group,state):
       
        L_x = state["L_x"]
        L_g = state["L_g"]
        Q_x = state["Q_x"]
        Q_g = state["Q_g"]
        
        g = weight.grad.data
        s = g.shape
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)
        
        g = torch.mm(torch.mm(Q_g.t(), g), Q_x)
        #g = torch.mm(torch.mm(torch.diag((L_g+self.damping)**(-1)), g),torch.diag((L_x+self.damping)**(-1)) )
        g = g/(L_g.unsqueeze(1)*L_x.unsqueeze(0))
        g = torch.mm(torch.mm(Q_g, g), Q_x.t())
        
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb
    
    def precond_two_level(self):
        
        fisher_norm = 0
        
        y = self.compute_Fv_Ru()
        
        if self._iteration_counter % self.T_cov == 0:
            
            if self._iteration_counter==0:
                
                if not self.krylov:
                    
                    self.f_coarse = self.compute_f_coarse()
                    
                else:
                    
                    self.f_coarse = self.compute_f_coarse_krylov()
            else:
                
                rho = min(1-1/self._iteration_counter,self.alpha)
                
                if not self.krylov:
                    
                    self.f_coarse = rho*self.f_coarse+(1-rho)*self.compute_f_coarse()

                else:
                    
                    self.f_coarse = rho*self.f_coarse+(1-rho)*self.compute_f_coarse_krylov()

            self.f_coarse_inv = torch.inverse(self.f_coarse)
            
        y = torch.mm(self.f_coarse_inv,y).contiguous().view(-1)
        
        #Check error
        '''
        error = self.check_error(f_coarse=self.f_coarse,beta=y)
        
        print(f"Iteration: {self._iteration_counter}, error: {error}")
        
        self.errors.append(error.cpu().numpy())
        
        self.iterations.append(self._iteration_counter)
        '''
        
        start = 0
        
        if not self.krylov:
            
            for iter,group in enumerate(self.param_groups):
                # Getting parameters

                if len(group['params']) == 2:
                    
                    weight, bias = group['params']
                    
                    state = self.state[weight]

                    s_w = weight.grad.shape
                    
                    d = np.prod(s_w)
                    
                    s_b = bias.grad.shape
                    
                    d_b = np.prod(s_b)
                    
                    r0 = state["r0"]
                    
                    r_b = r0[:, -1].contiguous().view(*s_b)
                    
                    weight.grad.data+= (y[iter]*r0[:, :-1]).contiguous().view(*s_w)
                    
                    bias.grad.data+=(y[iter]*r_b)

                    if self.constraint_norm:
                        
                        fisher_norm += float(torch.abs((weight.grad*self.flat_grads[start:start+d].view(*s_w)).sum()))
                        
                        fisher_norm += float(torch.abs((bias.grad*self.flat_grads[start+d:start+d+d_b].view(*s_b)).sum()))
                        
                        start+=d+d_b

                else:
                    weight = group['params'][0]
                    
                    state = self.state[weight]
                    
                    s = weight.grad.shape
                    
                    weight.grad.data+=(y[iter]*state["r0"]).contiguous().view(*s)
                    
                    d = np.prod(s)
                    
                    if self.constraint_norm:
                        
                        fisher_norm += float(torch.abs((weight.grad * self.flat_grads[start:start+d].view(*s)).sum()))
                        
                        start+=d

        else:
            
            for i,group in zip(range(0,2*self.num_layer,2),self.param_groups):
                
                # Getting parameters
                if len(group['params']) == 2:
                    
                    weight, bias = group['params']
                    
                    state = self.state[weight]
                    
                    s_w = weight.grad.shape
                    
                    d = np.prod(s_w)
                    
                    s_b = bias.grad.shape
                    
                    d_b = np.prod(s_b)
                    
                    weight.grad.data+= (y[i]*state["r1"][:, :-1]+y[i+1]*state["r2"][:, :-1]).contiguous().view(*s_w)
                    
                    bias.grad.data+=(y[i]*state["r1"][:, -1]+y[i+1]*state["r2"][:, -1]).contiguous().view(*s_b)

                    if self.constraint_norm:
                        
                        fisher_norm += float(torch.abs((weight.grad*self.flat_grads[start:start+d].view(*s_w)).sum()))
                        
                        fisher_norm += float(torch.abs((bias.grad*self.flat_grads[start+d:start+d+d_b].view(*s_b)).sum()))
                        
                        start+=d+d_b

                else:
                    
                    weight = group['params'][0]
                    
                    state = self.state[weight]
                    
                    s = weight.grad.shape
                    
                    d = np.prod(s)
                    
                    weight.grad.data+=(y[i]*state["r1"]+y[i+1]*state["r2"]).contiguous().view(*s)
                    
                    if self.constraint_norm:
                        
                        fisher_norm += float(torch.abs((weight.grad * self.flat_grads[start:start+d].view(*s)).sum()))
                        
                        start+=d
        
        return fisher_norm 

        
    
    
    def gather_list_grad(self):
        
        """Gather all the gradients of the networks and return a vector containing them."""
        
        grads = []
        
        for group in self.param_groups:
            
            if len(group['params']) == 2:
                
                weight, bias = group['params']
                
            else:
                
                weight = group['params'][0]
                
                bias = None
            
            g = weight.grad.data
            
            grads.append(g.contiguous().view(-1))
            
            if bias is not None:
                
                gb = bias.grad.data
                
                grads.append(gb.contiguous().view(-1))
            
            
        
        return torch.cat(grads,0)

    def check_error(self,f_coarse,beta):
        """
        Check the error made by two-level methods compared to one-level KFAC
        """
        
        #print(f"betea: {torch.norm(beta,p=2)}")
        
        first_term = torch.dot(f_coarse@beta,beta)
        
        second_term = 0
        
        if not self.krylov:
            
            for iter,group in enumerate(self.param_groups):
                
                if len(group['params']) == 2:
                    
                    weight, bias = group['params']
                    
                else:
                    
                    weight = group['params'][0]
                
                state = self.state[weight]
                
                    
                second_term+=(beta[iter]*state["r0"]*state["residual"]).sum()
                    
                
        else:
            
            for i,group in zip(range(0,2*self.num_layer,2),self.param_groups):

                if len(group['params']) == 2:
                    
                    weight, bias = group['params']
                    
                else:
                    
                    weight = group['params'][0]

                state = self.state[weight]
                    
                second_term+= ((beta[i]*state["r1"]+beta[i+1]*state["r2"])*state["residual"]).sum()
                    

        error = first_term-2*second_term
        #print(f"\tFisrt term :{first_term}")
        #print(f"\tSecond term :{second_term}")
        return error
    
    
    def _compute_covs(self, group, state,it):
        
        """Computes the covariances."""

        
        x = state["x"]
        
        gy = state["gy"]
        
        T = state["num_locations"] 
        
        mod = group['mod']
        
        if group["layer_type"]=="Conv2d":
            
            r = min(mod.kernel_size)-1
            
        self.batch_size = int(x.shape[1]/T)
        
        #print(f"batch size: {self.batch_size}")
        
        #print(f"Layer: {it}")
        
        if self._iteration_counter == 0:
            
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])

            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
    
                
        else: #self._iteration_counter > 0
            
            rho = min(1-1/self._iteration_counter,self.alpha)
            
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                            beta=rho,
                            alpha=(1-rho) /float(gy.shape[1]))

            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                            beta=rho,
                            alpha=(1-rho)/float(x.shape[1]))

    def _inv_covs(self, xxt, ggt, num_locations):
        
        """Inverses the covariances."""
        
        # Computes pi
        pi = 1.0
        
        if self.pi:
            
            tx = torch.trace(xxt) * ggt.shape[0]
            
            tg = torch.trace(ggt) * xxt.shape[0]
            
            pi = (torch.abs(tx / tg))**0.5
            
        # Regularizes and inverse
        x_add = pi*(self.damping)**0.5
        
        g_add = (1.0/pi)*(self.damping)**0.5
        
        diag_xxt = xxt.new(xxt.shape[0]).fill_(x_add)
        
        diag_ggt = ggt.new(ggt.shape[0]).fill_(g_add)
        
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        
        return ixxt, iggt
    
    
    def multiply_by_inverse(self,state,vector):
        
        L_x = state["L_x"]
        
        L_g = state["L_g"]
        
        Q_x = state["Q_x"]
        
        Q_g = state["Q_g"]
        
        s = (state['ggt'].shape[1],state['xxt'].shape[0])
        
        vector = torch.mm(torch.mm(Q_g.t(), vector.contiguous().view(*s)), Q_x)
        
        vector = vector/((L_g+self.damping).unsqueeze(1)*(L_x+self.damping).unsqueeze(0))
        
        vector = torch.mm(torch.mm(Q_g, vector), Q_x.t())
        
        del L_x
        
        del L_g
        
        del Q_x
        
        del Q_g
        #torch.cuda.empty_cache() 
        
        return vector
        
    
    def compute_coarse_space(self,group,residual):
        
        """Compute the coarse space"""
        
        weight = group['params'][0]
        
        state = self.state[weight]
        
        shape = residual.shape
        
        if 'r0' in state:
            
            del state["r0"]
        
        if self.coarse_space == "nicolaides":
            
            r0 = torch.ones_like(residual,device=self.device)
        
        elif self.coarse_space=="residual":
            
            r0 = self.multiply_by_inverse(state,residual)
            
        elif self.coarse_space=="tpselepedis":
        
            r0 = torch.ones_like(residual,device=self.device)
        
        elif self.coarse_space=="spectral":
            
            Q_x = state["Q_x"]
            
            Q_g = state["Q_g"]
            
            r0 = torch.kron(Q_x[:,0],Q_g[:,0])
        
        else:
            
            raise ValueError("Unvalid coarse space. Choose the coarse space from the following list \
                             [nicolaides,spectral,residual,tpselepedis]")
            
        state["r0"] = r0.contiguous().view(*shape)/(torch.norm(r0,p=np.inf)+1e-8)
        
        
    def compute_coarse_space_krylov(self,group,residual):
        
        """Compute the coarse space in the Krylov case."""
        
        weight = group['params'][0]
        
        state = self.state[weight]
        
        shape = residual.shape
        
        if 'r1' in state:
            
            del state["r1"]
         
        if 'r2' in state:
            
            del state["r2"]
            
        if self.coarse_space=="nicolaides":
            
            r1 = torch.ones_like(residual,device=self.device)
            
        elif self.coarse_space=="residual":
    
            r1 = self.multiply_by_inverse(state,residual)
        
        elif self.coarse_space in ["spectral","tpselepedis"]:
        
            raise ValueError("Can not implement krylov coarse space for this method. Please choose nicolaides or residual.")
        
        r2 = self.multiply_by_inverse(state,r1)

        state["r1"] = r1.contiguous().view(*shape)/(torch.norm(r1,p=np.inf)+1e-8)
        
        state["r2"] = r2.contiguous().view(*shape)/(torch.norm(r2,p=np.inf)+1e-8)
         
        
    
    def collect_stats(self):
        
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
            
                
    def GTu_MLP(self,state,g):
        
        """compute Diag(G_i^TW_iA_{i-1}) for a MLP layer"""
    
        x = state["x"]
        
        gy = state["gy"]
        
        return torch.diagonal(torch.mm(torch.mm(gy.t(), g), x)).view(-1,1)
    
    
    def GTu_CNN(self,state,g):
        
        """compute Diag(G_i^TW_iA_{i-1}) for a CNN layer"""
        
        x = state["x"]
        
        gy = state["gy"]
        
        T = state["num_locations"] 
            
        D = torch.diagonal(torch.mm(torch.mm(gy.t(),g),x))
        
        r = [torch.sum(D[b*T:(b+1)*T]) for b in range(self.batch_size)]
        
        r = torch.tensor(r,device=self.device)
        
        return r.view(-1,1)
                            
    
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
            
            g = weight.grad.data
            
            s = g.shape
            
            if group['layer_type']=='Conv2d':
                
                g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
                
                if bias is not None:
                    
                    gb = bias.grad.data
                    
                    g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)
                    
                r+=self.GTu_CNN(state,g)
                
            elif group['layer_type']=='Linear':
                
                if bias is not None:
                    
                    gb = bias.grad.data
                    
                    g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)
                
                r+=self.GTu_MLP(state,g)
            
        return r
    
  
    
    def Fisher_vecteur_product_theta_kfac(self,state,group,w):
        
        if group['layer_type']=='Conv2d':
            
            r = self.Gv_CNN(state,w)/(self.batch_size)
            
        elif group['layer_type']=='Linear':
            
            r = self.Gv_MLP(state,w)/(self.batch_size)
    
        return r
    
    def compute_Fv_Ru(self):
        
        """
        Compute u=(grad-Fg) where grad is the gradient and g = F^{-1}_{approx}grad or u = grad if tpselepedis coarse space
        
        return R_0u where R_0 is the matrix of coarse space
        
        """
        
        
        results = []
        
        start = 0
        
        if self.coarse_space == "tpselepedis":
            
        #    w = self.GTu() #Just for checking error
            
            for iter,group in enumerate(self.param_groups):
                
                if len(group['params']) == 2:
                    
                    weight, bias = group['params']
                    
                else:
                    
                    weight = group['params'][0]
                    
                    bias = None
                    
                g = weight.grad.data
                
                state = self.state[weight]
                
                s = g.shape
                
                if group['layer_type'] == 'Conv2d':
                    
                    g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
                    
                if bias is not None:
                    
                    gb = bias.grad.data
                    
                    g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)
                    
        
                s = g.shape
            
                residual = torch.empty_like(g)
                
                d = np.prod(g.shape)
               
                if not self.krylov:
                    
                    self.compute_coarse_space(group,residual)
                    
                    results.append((state["r0"]*self.flat_grads[start:start+d].contiguous().view(*s)).sum())
                    
                else:
                    
                    self.compute_coarse_space_krylov(group,residual)
                    
                    results.append((state["r1"]*self.flat_grads[start:start+d].contiguous().view(*s)).sum())
                    
                    results.append((state["r2"]*self.flat_grads[start:start+d].contiguous().view(*s)).sum())

                # just for checking error
                #residual = self.flat_grads[start:start+d].contiguous().view(*s)-self.Fisher_vecteur_product_theta_kfac(state,group,w)-self.damping*g
                #state["residual"]=residual
                
                start+=d  
        else:
            
            w = self.GTu() 
            
            for iter,group in enumerate(self.param_groups):
                
                if len(group['params']) == 2:
                    
                    weight, bias = group['params']
                    
                else:
                    
                    weight = group['params'][0]
                    
                    bias = None
                
                g = weight.grad.data
                
                s = g.shape
                
                if group['layer_type'] == 'Conv2d':
                    
                    g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
                    
                if bias is not None:
                    
                    gb = bias.grad.data
                    
                    g = torch.cat([g, gb.contiguous().view(gb.shape[0], 1)], dim=1)
                
                state = self.state[weight]
                
                d = np.prod(g.shape)
                
                s = g.shape
                
                residual = self.flat_grads[start:start+d].contiguous().view(*s)-self.Fisher_vecteur_product_theta_kfac(state,group,w)-self.damping*g
                
                state["residual"]=residual
                
                if not self.krylov:
                    
                    self.compute_coarse_space(group,residual)
                    
                    results.append((state["r0"]*self.flat_grads[start:start+d].contiguous().view(*s)).sum())
                    
                else:
                    
                    self.compute_coarse_space_krylov(group,residual)
                    
                    results.append((state["r1"]*self.flat_grads[start:start+d].contiguous().view(*s)).sum())
                    
                    results.append((state["r2"]*self.flat_grads[start:start+d].contiguous().view(*s)).sum())

                start+=d
                
        results = torch.tensor(results,device=self.device)
        
        return results.contiguous().view(-1,1)
    

    
   
    
    def f_coarse_ij(self,group_i,group_j):
        
        """Compute en entry F_{i,j} of Fcoarse"""
        
        weight_i = group_i['params'][0]
        
        state_i = self.state[weight_i]
        
        weight_j = group_j['params'][0]
        
        state_j = self.state[weight_j]
        
        r0_i = state_i['r0']
        
        r0_j = state_j["r0"]
        
        if group_j['layer_type'] == 'Conv2d':
            
            v = self.GTu_CNN(state_j,r0_j)
            
        elif group_j['layer_type'] == 'Linear':
            
            v = self.GTu_MLP(state_j,r0_j)
        
        if group_j['layer_type'] == 'Conv2d':
            
            u = self.Gv_CNN(state_i,v)
            
        elif group_j['layer_type'] == 'Linear':
            
            u = self.Gv_MLP(state_i,v)
        
        result = (r0_i*u).sum()/self.batch_size
        
        return result
    
    def f_coarse_ij_tlepedis(self,group_i,group_j):
        
        """
        Compute en entry F_{i,j} of Fcoarse in the case 
        where the coarse space is tpselepedis
        """
        
        weight_i = group_i['params'][0]
        
        state_i = self.state[weight_i]
        
        weight_j = group_j['params'][0]
        
        state_j = self.state[weight_j]
        
        r0_i = state_i['r0']
        
        r0_j = state_j["r0"]
        
        x_i = state_i["x"]
        
        g_i = state_i["gy"]
        
        x_j = state_j["x"]
        
        g_j = state_j["gy"]
        
        T_i = state_i["num_locations"]
        
        T_j = state_j["num_locations"]
        
        if  group_i['layer_type'] == "Conv2d" or  group_j['layer_type']=="Conv2d":
            
            T = min(T_i,T_j)
            
            if T==T_i:
                
                x_j = x_j.contiguous().view(self.batch_size,-1,T_j)
                
                x_j = torch.nn.functional.interpolate(x_j, T, mode='nearest')
                
                g_j = g_j.contiguous().view(self.batch_size,-1,T_j)
                
                g_j = torch.nn.functional.interpolate(g_j, T, mode='nearest')
                
                x_j = x_j.contiguous().view(-1,self.batch_size*T)
                
                g_j = g_j.contiguous().view(-1,self.batch_size*T)
                
            elif T==T_j:
                x_i = x_i.contiguous().view(self.batch_size,-1,T_i)
                
                x_i = torch.nn.functional.interpolate(x_i, T, mode='nearest')
                
                g_i = g_i.contiguous().view(self.batch_size,-1,T_i)
                
                g_i = torch.nn.functional.interpolate(g_i, T, mode='nearest')
                
                x_i = x_i.contiguous().view(-1,self.batch_size*T)
                
                g_i = g_i.contiguous().view(-1,self.batch_size*T)
        
        A = torch.mm(x_i,x_j.t())/float(x_i.shape[1])
        
        G = torch.mm(g_i,g_j.t())/float(g_i.shape[1])
        
        r = (r0_i*torch.mm(G,torch.mm(r0_j,A.t()))).sum()
        
        return r

        

    def compute_f_coarse(self):
        
        """Compute matrix Fcoarse"""
        
        F = torch.zeros((self.num_layer,self.num_layer)).to(self.device)
        
        for i,group_i in enumerate(self.param_groups):
            
            for j,group_j in enumerate(self.param_groups[:i+1]):
                
                if i==j:
                    
                    weight = group_i['params'][0]
                    
                    state = self.state[weight]
                    
                    N_i = np.prod(state["r0"].shape)
                    
                    if self.coarse_space=="tpselepedis":
                        
                        F[i,i] = self.f_coarse_ij_tlepedis(group_i,group_i)+self.damping*N_i
                        
                    else:
                        
                        F[i,i] = self.f_coarse_ij(group_i,group_i)+self.damping*N_i
                else: 
                    
                    if self.coarse_space=="tpselepedis":
                        
                        F[i,j] = self.f_coarse_ij_tlepedis(group_i,group_j)
                        
                    else:
                        
                        F[i,j] = self.f_coarse_ij(group_i,group_j)
        
        return F+F.t()-torch.diag(torch.diagonal(F,offset=0))
    
    
    def compute_f_coarse_krylov(self):
        
        """Compute matrix Fcoarse in the krylov case"""
        
        F = torch.zeros((2*self.num_layer,2*self.num_layer)).to(self.device)
        
        def compute(state1,state2,r1,r2,group1,group2):
            
            if group2['layer_type'] == 'Conv2d':
                
                v = self.GTu_CNN(state2,r2)
                
            elif group2['layer_type'] == 'Linear':
                
                v = self.GTu_MLP(state2,r2)

            if group1['layer_type'] == 'Conv2d':
                
                u = self.Gv_CNN(state1,v)
                
            elif group1['layer_type'] == 'Linear':
                
                u = self.Gv_MLP(state1,v)

            return (r1*u).sum()/self.batch_size
    
        
        for i,group_i in zip(range(0,2*self.num_layer,2),self.param_groups):
            
            for j,group_j in zip(range(0,2*self.num_layer,2),self.param_groups):
                
                weight_i = group_i['params'][0]
                
                state_i = self.state[weight_i]
                
                weight_j = group_j['params'][0]
                
                state_j = self.state[weight_j]
                
                r1_i = state_i['r1']
                
                r2_i = state_i['r2']
                
                r1_j = state_j["r1"]
                
                r2_j = state_j["r2"]
                
                N_i = np.prod(r1_i.shape)
                
                N_j = np.prod(r2_j.shape)
                
                F[i,j] = compute(state_i,state_j,r1_i,r1_j,group_i,group_j) + (N_i*self.damping)*(i==j)
                
                F[i+1,j] = compute(state_i,state_j,r2_i,r1_j,group_i,group_j)
                
                F[i,j+1] = compute(state_i,state_j,r1_i,r2_j,group_i,group_j)
                
                F[i+1,j+1] = compute(state_i,state_j,r2_i,r2_j,group_i,group_j) + (N_i*self.damping)*(i==j)
                
                   
        return F

