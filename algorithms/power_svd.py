import torch
import numpy as np
from typing import Dict, List, Tuple, Union

from algorithms.utils import *


def power_svd_mlp(a:torch.tensor,g:torch.tensor,v:torch.tensor,A:torch.tensor
                  ,G:torch.tensor,epsilon:float,max_iter:int,method:str)-> Tuple[torch.tensor]:
    
    """
    Perform the power SVD algorithm for a mutlti-layer perceptron layer.
    
    Parameters:
    ----------
    a: torch.tensor
        Matrix of activations
    g: torch.tensor
        matrix of pre-activation derivatives
    v: torch.tensor
        Initial vector to start the algorithm
    A: torch.tensor or None
       Rank-1 term in two-rank approximations (first term in the Kronecker product)
    G: torch.tensor
      Rank-1 term in two-rank approximations (second term in the Kronecker product)  
    epsilon: float
        Precision of the power SVD algorithm
    max_iter: int
        Maximum number of iterations to perform in the power SVD algorithm
    method: str
    One of the three methoths between kpsvd,deflation and kfac_cor
    
    
    Returns
    --------
    R: torch.tensor
        matrix corresponding to the left singular vector
    S: torch.tensor
        matrix corresponding to the right singular vector
    """
    
    d_a = a.shape[1]
    d_g = g.shape[1]
    m = a.shape[0]
    a,g = a.detach().clone(),g.detach().clone()
    if A is not None:
        A_vec = vec(A)
        G_vec = vec(G)
    v = v/torch.norm(v,p=2)
    k=0
    error=1000
    while True:
        if k==0:
            w =  ZFv_mlp(v,d_g,a,g,m)
            if method in ["kfac_cor","deflation"]:
                w-=torch.dot(G_vec.squeeze(),v.squeeze())*A_vec
        else:
            w = w_k_plus.detach().clone()
        alpha = torch.norm(w,p=2)
        u = w/alpha
        z = ZFTu_mlp(u,d_a,a,g,m)
        if method in ["kfac_cor","deflation"]:
            z-=torch.dot(A_vec.squeeze(),u.squeeze())*G_vec
        beta = torch.norm(z,p=2)
        v = z/beta
        sigma = beta
        w_k_plus = ZFv_mlp(v,d_g,a,g,m)
        if method in ["kfac_cor","deflation"]:
            w_k_plus-=torch.dot(G_vec.squeeze(),v.squeeze())*A_vec 
        error = torch.norm(w_k_plus-sigma*u,p=2)
        k+=1
        if error<epsilon or k>max_iter:
            break
    R = (sigma**0.5)*u.contiguous().view((d_a,d_a))
    S = (sigma**0.5)*v.contiguous().view((d_g,d_g))
    return R.t(),S.t()

    
    
def power_svd_cnn(a:torch.tensor,g:torch.tensor,V:torch.tensor,A:torch.tensor,
                  G:torch.tensor,epsilon:float,max_iter:int,r:int,method:str)->Tuple[torch.tensor]:
    
    """
    Perform the power SVD algorithm for a convolutional layer.
    
    Parameters:
    ----------
    a: torch.tensor
        Matrix of activations
    g: torch.tensor
        matrix of pre-activation derivatives
    V: torch.tensor
        Matrix corresponding to initial vector used to start the algorithm
    A: torch.tensor or None
       Rank-1 term in two-rank approximations (first term in the Kronecker product)
    G: torch.tensor
      Rank-1 term in two-rank approximations (second term in the Kronecker product)  
    epsilon: float
        Precision of the power SVD algorithm
    max_iter: int
        Maximum number of iterations to perform in the power SVD algorithm
    r:int
        Radius of the area used to approximate Z(F)v and Z(F)^Tu
    method: str
    One of the three methoths between kpsvd,deflation and kfac_cor
    
    
    Returns
    --------
    R: torch.tensor
        matrix corresponding to the left singular vector
    S: torch.tensor
        matrix corresponding to the right singular vector
    """
    
   
    V/= torch.sqrt((V*V).sum()) # normalization v = v/||v||
    k = 0
    error = 1000
    while True:
        if k==0: 
            W = ZFv_cnn(a,g,V,r) # w = Z(F)v
            if method in ["kfac_cor","deflation"]:
                W-= ((G*V).sum())*A # w-= <vec(G),v>A
        else:
            W = W_k_plus.detach().clone()
        alpha = torch.sqrt((W**2).sum()) # norm of W
        U = W/alpha
        Z = ZFTu_cnn(a,g,U,r) # Z(F)^Tu
        if method in ["kfac_cor","deflation"]:  
            Z-=  ((A*U).sum())*G # z = z - <vec(A),u>G 
        beta = torch.sqrt((Z**2).sum()) # norm of z
        V = Z/beta
        sigma = beta
        W_k_plus =  ZFv_cnn(a,g,V,r) # w = Z(F)v
        if method in ["kfac_cor","deflation"]:
            W_k_plus-= ((G*V).sum())*A # w-= <vec(G),v>A
        error = torch.sqrt(((W_k_plus-sigma*U)**2).sum())
        k+=1
        if error<epsilon or k>max_iter:  
            break
    return (sigma**0.5)*U,(sigma**0.5)*V
    