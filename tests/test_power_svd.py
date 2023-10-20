import torch
from algorithms.power_svd import power_svd_mlp,power_svd_cnn


def test_power_svd_mlp():
    d_a = 20
    d_g = 30
    m = 256
    device = torch.device("cuda:0")
    a = torch.randn((m,d_a)).to(device)
    g = torch.randn((m,d_g)).to(device)
    v = torch.randn((d_g**2,1)).to(device)
    A = torch.mm(a.t(),a)/m
    G = torch.mm(g.t(),g)/m
    epsilon = 1e-6
    max_iter = 1000
    
    R_kpsvd, S_kpsvd = power_svd_mlp(a=a,g=g,v=v,A=None,G=None,
                                     epsilon=epsilon,max_iter=max_iter,method="kpsvd")
    R_kfac_cor, S_kfac_cor = power_svd_mlp(a=a,g=g,v=v,A=A,G=G,
                                     epsilon=epsilon,max_iter=max_iter,method="kfac_cor")
    R_deflation, S_deflation = power_svd_mlp(a=a,g=g,v=v,A=A,G=G,
                                     epsilon=epsilon,max_iter=max_iter,method="deflation")
    
    assert  R_kpsvd.shape == (d_a,d_a)
    assert S_kpsvd.shape == (d_g,d_g)
    assert  R_kfac_cor.shape == (d_a,d_a)
    assert S_kfac_cor.shape == (d_g,d_g)
    assert  R_deflation.shape == (d_a,d_a)
    assert S_deflation.shape == (d_g,d_g)
    


def test_power_svd_cnn():
    d_a = 20
    d_g = 30
    m = 256
    T = 27*27
    device = torch.device("cuda:0")
    a = torch.randn((m,d_a,T)).to(device)
    g = torch.randn((m,d_g,T)).to(device)
    V = torch.randn((d_g,d_g)).to(device)
    A = a.data.permute(1, 0, 2).contiguous().view(a.shape[1], -1)
    G = g.data.permute(1, 0, 2).contiguous().view(g.shape[1], -1)
    A = torch.mm(A,A.t())/a.shape[1]
    G = torch.mm(G,G.t())/g.shape[1]
    epsilon = 1e-6
    max_iter = 1000
    r=3
    R_kpsvd, S_kpsvd = power_svd_cnn(a=a,g=g,V=V,A=None,
                  G=None,epsilon=epsilon,max_iter=max_iter,r=r,method="kpsvd")
    
    R_kfac_cor, S_kfac_cor = power_svd_cnn(a=a,g=g,V=V,A=A,
                  G=G,epsilon=epsilon,max_iter=max_iter,r=r,method="kfac_cor")
    
    R_deflation, S_deflation = power_svd_cnn(a=a,g=g,V=V,A=A,
                  G=G,epsilon=epsilon,max_iter=max_iter,r=r,method="deflation")
    
    assert  R_kpsvd.shape == (d_a,d_a)
    assert S_kpsvd.shape == (d_g,d_g)
    assert  R_kfac_cor.shape == (d_a,d_a)
    assert S_kfac_cor.shape == (d_g,d_g)
    assert  R_deflation.shape == (d_a,d_a)
    assert S_deflation.shape == (d_g,d_g)    
    
    
    
    
    
    