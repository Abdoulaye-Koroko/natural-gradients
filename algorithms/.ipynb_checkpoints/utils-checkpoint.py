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

