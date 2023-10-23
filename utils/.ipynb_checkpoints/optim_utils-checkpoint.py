import torch
import numpy as np
from typing import Dict, List, Tuple, Union

def vec(mat:torch.tensor)-> torch.tensor:
    """
    Transforms a matrix into a vecteur.
    
    Parameters
    ----------
    mat: torch.tensor
        Matrix to vectorize
    
    Returns
    -------
    x: torch.tensor
        result of the operation
    """
    
    x = torch.unsqueeze(mat.transpose(1, 0).flatten(),1)
    return x


def ZFv_mlp(v:torch.tensor,d_g:int,a:torch.tensor,g:torch.tensor,m:int)->torch.tensor:
    """
    Compute the operation Z(F)v for a multi-layer perceptron layer.
    
    Parameters:
    ----------
    v: torch.tensor
        Vector with which we perform the matrix-vector operation
    d_g: int
        Dimension of pre-activation derivatives in the layer
    a: torch.tensor
        Matrix of activations
    g: torch.tensor 
        Matrix of pre-activation derivatives
    m: int
        Batch size
    
    Returns
    -------
    res: torch.tensor
        Result of the matrix-vector multiplication
        
    
    """
    
    V = v.contiguous().view((d_g,d_g)).t()
    G = torch.squeeze(torch.bmm(torch.mm(g,V).contiguous().view(g.shape[0], 1, g.shape[1]),
                                g.contiguous().view(g.shape[0], g.shape[1], 1)),dim=2)
    sign = torch.zeros_like(G)+1
    sign[G<0]=-1
    G = (torch.abs(G))**0.5
    a = a*G.expand_as(a)
    a2 = a*sign.expand_as(a)
    A = (1/m)*a.t()@a2
    res = vec(A)
    return res


def ZFTu_mlp(u:torch.tensor,d_a:int,a:torch.tensor,g:torch.tensor,m:int):
    
    """
    Compute the operation Z(F)^Tu for a multi-layer perceptron layer.
    
    Parameters:
    ----------
    u: torch.tensor
        Vector with which we perform the matrix-vector operation
    d_a: int
        Dimension of activations in the layer
    a: torch.tensor
        Matrix of activations
    g: torch.tensor 
        Matrix of pre-activation derivatives
    m: int
        Batch size
    
    Returns
    -------
    res: torch.tensor
        Result of the matrix-vector multiplication
        
    
    """
    
    U = u.contiguous().view((d_a,d_a)).t()
    A = torch.squeeze(torch.bmm(torch.mm(a,U).contiguous().view(a.shape[0], 1,
                    a.shape[1]), a.contiguous().view(a.shape[0], a.shape[1], 1)),dim=2)
    sign = torch.zeros_like(A)+1
    sign[A<0]=-1.0
    A = (torch.abs(A))**0.5
    g = g*A.expand_as(g)
    g2 = g*sign.expand_as(g)
    G = (1/m)*g.t()@g2
    res = vec(G)
    return res



def ZFv_cnn(a:torch.tensor,g:torch.tensor,V:torch.tensor,r:int)->torch.tensor:
    """
    Compute the operation Z(F)v for a convolutional layer.
    
    Parameters
    ----------
    a: torch.tensor
        Matrix of activations
    g: torch.tensor
        Matrix of pre-activation derivatives
    V: torch.tensor
         Matrix corresponding to  the vector with which we perform the matrix-vector operation
    r: int
        Radius of the area used to approximate Z(F)v
        
    Returns
    -------
    res: torch.tensor
        Matrix corresponding to the result of Z(F)v
    """
    
    batch_size,d,T = a.shape
    t = torch.tensor(range(T))
    indexes = torch.cartesian_prod(t,t)
    num_locs = len(indexes)
    gap = torch.abs(indexes[:,0]-indexes[:,1])
    indexes =  indexes[gap<=r]
    K = torch.bmm(torch.bmm(g.data.permute(0,2,1),V.view(1,V.shape[0],V.shape[1]).expand(batch_size,-1,-1)),g)
    K = K.view(batch_size,-1)[:,gap<=r]
    K = K.view(batch_size,1,-1).expand(-1,d,-1)
    a_t = a.data.permute(0,2,1)[:,indexes[:,0]]
    a_tpr = a.data.permute(0,2,1)[:,indexes[:,1]]
    res = torch.mean(torch.bmm(K*a_t.data.permute(0,2,1),a_tpr),dim=0)
    
    return res



def ZFTu_cnn(a:torch.tensor,g:torch.tensor,U:torch.tensor,r:int)->torch.tensor:
    
    """
    Compute the operation Z(F)^Tu for a convolutional layer.
    
    Parameters
    ----------
    a: torch.tensor
        Matrix of activations
    g: torch.tensor
        Matrix of pre-activation derivatives
    U: torch.tensor
         Matrix corresponding to  the vector with which we perform the matrix-vector operation
    r: int
        Radius of the area used to approximate Z(F)^Tu
        
    Returns
    -------
    res: torch.tensor
        Matrix corresponding to the result of Z(F)^Tu
    """
    
    batch_size,d,T = g.shape
    t = torch.tensor(range(T))
    indexes = torch.cartesian_prod(t,t)
    num_locs = len(indexes)
    gap = torch.abs(indexes[:,0]-indexes[:,1])
    indexes =  indexes[gap<=r]
    K = torch.bmm(torch.bmm(a.data.permute(0,2,1),U.view(1,U.shape[0],U.shape[1]).expand(batch_size,-1,-1)),a)
    K = K.view(batch_size,-1)[:,gap<=r]
    K = K.view(batch_size,1,-1).expand(-1,d,-1)
    g_t = g.data.permute(0,2,1)[:,indexes[:,0]]
    g_tpr = g.data.permute(0,2,1)[:,indexes[:,1]]
    res = torch.mean(torch.bmm(K*g_t.data.permute(0,2,1),g_tpr),dim=0)
    
    return res



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
   
