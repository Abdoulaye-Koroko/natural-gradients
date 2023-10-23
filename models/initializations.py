import torch
import torch.nn as nn

seed = 0

def xavier_uniform_init_weights(m):
    torch.manual_seed(seed)
    if type(m) == nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def sparse_init_weights(m):
    torch.manual_seed(seed)
    if type(m) == nn.Linear or type(m)==nn.Conv2d:
        nn.init.sparse_(m.weight,std=1, sparsity=0.8)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def xavier_normal_init_weights(m):
    torch.manual_seed(seed)
    if type(m) == nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def normal_init_weights(m):
    torch.manual_seed(seed)
    if type(m) == nn.Linear or type(m)==nn.Conv2d:
        nn.init.normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)