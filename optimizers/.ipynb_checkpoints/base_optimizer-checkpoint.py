from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F




class BaseOptimizer(ABC,Optimizer):
    """
    This abstract class will serve as a base class for implementing all the optimizer
    
    """
    
    def __init__(self):
        super().__init__([torch.zeros((1,1))], {})
    
    
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
        """Inverses the covariances."""
    
    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
    