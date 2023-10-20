from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F




class BaseOptimizer(ABC,Optimizer):
    """
    This abstract class will serve as a base class for implementing all the optimizer
    
    """
    
    def __init__(self,net,damping=1e-3,pi=False, T_cov=100,T_inv=100,
                 alpha=0.95, constraint_norm=False,kl=1e-2,batch_size=64):
    
        """
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
        self.damping = damping
        self.pi = pi
        self.T_cov = T_cov
        self.T_inv = T_inv
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        self.kl = kl
        self.update_stats = True
        self.batch_size = batch_size

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

        super().__init__(self.params, {})
    
    
    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training and self.update_stats and self._iteration_counter%self.T_cov==0:
            self.state[mod]['x'] = i[0]


    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training and self.update_stats and self._iteration_counter%self.T_cov==0:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)
            
    @abstractmethod        
    def step(self, update_params=True):
        """Performs one step of preconditioning."""
    
    @abstractmethod
    def _precond(self, weight, bias, group, state):
        """Applies preconditioning.""" 
        
    @abstractmethod
    def _compute_covs(self, group, state):
        """Compute the covariance matrices (i.e. Kronecker factors)"""
    
    @abstractmethod
    def _inv_covs(self, xxt, ggt, num_locations):
        """ Compute inverses the covariance matrices."""
    
    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
    