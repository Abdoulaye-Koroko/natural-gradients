import torch
from utils.optim_utils import lanczos_mlp, lanczos_cnn


def test_lanczos_mlp():
    d_a = 20
    d_g = 30
    m = 256
    device = torch.device("cuda:0")
    a = torch.randn((m,d_a)).to(device)
    g = torch.randn((m,d_g)).to(device)
    q = torch.randn((d_g**2,1)).to(device)
    K = 10
    epsilon = 1e-6
    R,S,P,Q = lanczos_mlp(a,g,q,K,epsilon)
    
    assert R.shape==(d_a,d_a)
    assert S.shape==(d_g,d_g)
    assert P.shape==(d_a,d_a)
    assert Q.shape==(d_g,d_g)
    

def test_lanczos_cnn():
    d_a = 20
    d_g = 30
    m = 256
    T = 27*27
    device = torch.device("cuda:0")
    a = torch.randn((m,d_a,T)).to(device)
    g = torch.randn((m,d_g,T)).to(device)
    q = torch.randn((d_g**2,1)).to(device)
    epsilon = 1e-6
    r=3
    K = 10
    epsilon = 1e-6
    R,S,P,Q = lanczos_cnn(a,g,q,K,epsilon,r)
    
    assert R.shape==(d_a,d_a)
    assert S.shape==(d_g,d_g)
    assert P.shape==(d_a,d_a)
    assert Q.shape==(d_g,d_g)
    