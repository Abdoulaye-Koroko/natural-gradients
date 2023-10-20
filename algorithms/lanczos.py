import torch
import numpy as np
from typing import Dict, List, Tuple, Union

from algorithms.utils import *

def lanczos_mlp(a:torch.tensor,g:torch.tensor,q:torch.tensor,K:int,epsilon:float)->Tuple[torch.tensor]:
    """
    Perform the Lanczos algorithm for a multi-layer perceptron layer.
    
    Parameters
    ----------
    a: torch.tensor
        Matrix of activations
    g: torch.tensor
        Matrix of pre-activation derivatives
    q: torch.tensor
        Initial vector used to start the algorithm
    K:int
        Dimension of Krylov subspace
    epsilon:float
        Precison for stopping the algorithm
        
    Returns
    -------
    R: torch.tensor
        Matrix corresping to left singular vector associated to the greatest singular value
    S: torch.tensor
        Matrix corresping to right singular vector associated to the greatest singular value
    P: torch.tensor
        Matrix corresping to left singular vector associated to the second singular value.
    Q: torch.tensor
        Matrix corresping to right singular vector associated to the second singular value.
        
    
    """
    device = a.get_device()
    if device<0:
        device = torch.device('cpu')
    n_eigenvecs = 2
    d_a = a.shape[1]
    d_g = g.shape[1]
    m = a.shape[0]
    a,g = a.detach().clone(),g.detach().clone()
    def iteration(q):
        q = q/torch.norm(q,p=2)
        H = torch.zeros((K,K),device=device)
        P = torch.zeros((d_a*d_a,K),device=device)
        Q = torch.zeros((d_g*d_g,K),device=device)
        w = ZFv_mlp(q,d_g,a,g,m).squeeze()
        alpha = torch.norm(w,p=2)
        P[:,0]= w/alpha
        H[0,0] = alpha
        Q[:,0] = q.squeeze()
        for k in range(K-1):
            p = P[:,k]
            q = Q[:,k]
            alpha = H[k,k]
            z =  ZFTu_mlp(p,d_a,a,g,m)
            z-=alpha*q.unsqueeze(dim=1)
            beta = torch.norm(z,p=2)
            q =  z/beta
            Q[:,k+1] = q.squeeze()
            w = ZFv_mlp(q,d_g,a,g,m)
            w-=beta*p.unsqueeze(dim=1)
            alpha = torch.norm(w,p=2)
            P[:,k+1] = w.squeeze()/alpha
            H[k+1,k+1] = alpha
            H[k,k+1] = beta
            if beta<=epsilon or k>=K-1:
                break
        U_H,S,V_H = torch.svd(H)
        U_H,S,V_H = U_H[:,:n_eigenvecs],S[:n_eigenvecs],V_H[:,:n_eigenvecs]
        U = P@U_H
        V = Q@V_H
        
        return U,V,S,beta
    
    restart = 0
    while True:
        U,V,S,beta = iteration(q)
        sigma_1 = S[0]
        sigma_2 = S[1]
        error = torch.norm(ZFv_mlp(V[:,0],d_g,a,g,m).squeeze()-sigma_1*U[:,0],p=2)+ \
        torch.norm(ZFv_mlp(V[:,1],d_g,a,g,m).squeeze()-sigma_2*U[:,1],p=2)
        if error<=epsilon or restart>=100:
            break
        q = torch.sum(V,1)
        restart+=1
    
    sigma_1 = S[0]
    sigma_2 = S[1]
    R = (sigma_1**0.5)*U[:,0].contiguous().view((d_a,d_a))
    S = (sigma_1**0.5)*V[:,0].contiguous().view((d_g,d_g))
    P = (sigma_2**0.5)*U[:,1].contiguous().view((d_a,d_a))
    Q = (sigma_2**0.5)*V[:,1].contiguous().view((d_g,d_g))
    
    return R.t(),S.t(),P.t(),Q.t()






def lanczos_cnn(a:torch.tensor,g:torch.tensor,q:torch.tensor,K:int,epsilon:float,r:int)->Tuple[torch.tensor]:
    
    """
    Perform the Lanczos algorithm for a convolutional layer.
    
    Parameters
    ----------
    a: torch.tensor
        Matrix of activations
    g: torch.tensor
        Matrix of pre-activation derivatives
    q: torch.tensor
        Initial vector used to start the algorithm
    K:int
        Dimension of Krylov subspace
    epsilon:float
        Precison for stopping the algorithm
        
    Returns
    -------
    R: torch.tensor
        Matrix corresping to left singular vector associated to the greatest singular value
    S: torch.tensor
        Matrix corresping to right singular vector associated to the greatest singular value
    P: torch.tensor
        Matrix corresping to left singular vector associated to the second singular value.
    Q: torch.tensor
        Matrix corresping to right singular vector associated to the second singular value.
    
    """
    device = a.get_device()
    if device<0:
        device = torch.device('cpu')
    n_eigenvecs = 2
    d_a = a.shape[1]
    d_g = g.shape[1]
    restart_max = 10
    
    def iteration(q):
        q = q/torch.norm(q,p=2)
        H = torch.zeros((K,K),device=device)
        P = torch.zeros((d_a*d_a,K),device=device)
        Q = torch.zeros((d_g*d_g,K),device=device)
        w = vec(ZFv_cnn(a,g,q.view(d_g,d_g).t(),r)).squeeze()
        alpha = torch.norm(w,p=2)
        P[:,0]= w/alpha
        H[0,0] = alpha
        Q[:,0] = q.squeeze()
        for k in range(K-1):
            p = P[:,k]
            q = Q[:,k]
            alpha = H[k,k]
            z = vec(ZFTu_cnn(a,g,p.view(d_a,d_a).t(),r))
            z-=alpha*q.unsqueeze(dim=1)
            beta = torch.norm(z,p=2)
            q =  z/beta
            Q[:,k+1] = q.squeeze()
            w = vec(ZFv_cnn(a,g,q.view(d_g,d_g).t(),r))
            w-=beta*p.unsqueeze(dim=1)
            alpha = torch.norm(w,p=2)
            P[:,k+1] = w.squeeze()/alpha
            H[k+1,k+1] = alpha
            H[k,k+1] = beta
            if beta<=epsilon or k>=K-1:
                break
        
        U_H,S,V_H = torch.svd(H)
        U_H,S,V_H = U_H[:,:n_eigenvecs],S[:n_eigenvecs],V_H[:,:n_eigenvecs]
        U = P@U_H
        V = Q@V_H
        return U,V,S,beta
    
    restart = 0
    while True:
        U,V,S,beta = iteration(q)
        sigma_1 = S[0]
        sigma_2 = S[1]
        error = torch.norm(vec(ZFv_cnn(a,g,V[:,0].view(d_g,d_g).t(),r)).squeeze()-sigma_1*U[:,0],p=2)\
        +torch.norm(vec(ZFv_cnn(a,g,V[:,1].view(d_g,d_g).t(),r)).squeeze()-sigma_2*U[:,1],p=2)
        if error<=epsilon or restart>=restart_max:
            break
        q = torch.sum(V,1)
        restart+=1
        
        
    sigma_1 = S[0]
    sigma_2 = S[1]
    R = (sigma_1**0.5)*U[:,0].contiguous().view((d_a,d_a))
    S = (sigma_1**0.5)*V[:,0].contiguous().view((d_g,d_g))
    P = (sigma_2**0.5)*U[:,1].contiguous().view((d_a,d_a))
    Q = (sigma_2**0.5)*V[:,1].contiguous().view((d_g,d_g))
    return R.t(),S.t(),P.t(),Q.t()
   
