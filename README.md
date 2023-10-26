# Natural gradient-based optimization methods for deep learning

## Overview

This repository provides implementation in pytorch of several natural gradient-based optimization methods for training deep neural networks. It include the following methods (see in references section for more on each method)

- [x] KFAC
- [x] Two-level KFAC
- [x] KPSVD
- [x] Deflation
- [x] KFAC Corrected
- [x] Lanczos
- [x] Exact natural gratient with full Fisher matrix or with block-diagonal Fisher matrix


Author: Abdoulaye Koroko

- Email: abdoulayekoroko@gmail.com

- Github: [Abdoulaye-Koroko](https://github.com/Abdoulaye-Koroko)

## Installation


Clone this repository:

```sh
$ https://gitlab.ifpen.fr/supercalcul/natural-gradients.git
$ cd natural-gradients

```

Then, create a new virtual environment and install all the required packages:

```sh
$ pip install -e .
$ pip install -r requirements.txt
```

## Usage

You can use the optimizers developed to train several types of deep neural networks. Below are the different types of network with compatible optimizers

- Multi-layer perceptrons (MLP) : all optimizer

- Convolutional neural networks (CNN): all optimizer

- Deep convolutional auto-encoder (contains transposed convolutional layers): KFAC

- Deep convolutional GANs (contains transposed convolutional layers) : KFAC


### General use case

To use one of the proposed natural-gradient methods, you can write your training function as follows:

```python 
#First import the preconditioner of interest as bellow
from optimizers.kfac import KFAC
from optimizers.kpsvd import KPSVD
from optimizers.deflation import Deflation
from optimizers.kfac_cor import KFAC_CORRECTED
from optimizers.lanczos import Lanczos
from optimizers.twolevel_kfac import TwolevelKFAC
from optimizers.exact_natural_gradient import ExactNG

# Define your model
model = Mymodel()

#Define the optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,nesterov=False,weight_decay=weight_decay)

preconditioner = KFAC(model) # It can be any of the imported preconditioner above

#Define your dataloader
dataset = Mydata()

dataloader =  torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True,drop_last=True)
#Define your loss function
criterion = Myloss()

#device
device = torch.device("cuda")

# Training loop
for iter,batch in enumerate(trainloader):

    optimizer.zero_grad()
    
    inputs,labels=batch
    
    inputs,labels = inputs.to(device),labels.to(device)
    
    outputs = model(inputs)
    
    with torch.set_grad_enabled(True):
        
        loss = criterion(outputs,labels)
        
        preconditioner.update_stats = True
        
        loss.backward()
        
        preconditioner.step(update_params=True) 
    
        optimizer.step()
```


### MLP and CNN

You can train MLP or CNN networks with any optimizer using the `train_mlp.py` and `train_cnn.py` functions defined in the apps/mlp_cnn folder `apps/mlp_cnn`.

For example, to train the MLP deep auto-encoder with the CURVES dataset with KFAC optimizer, you just need to run the following command:

```
$ python apps/mlp_cnn/train_mlp.py --data CURVES --optim kfac --lr 1e-4 --damping 1e-4

```

All the default parameters of the functions `train_mlp.py` and `train_cnn.py` can be changed via `--` option. Below are different important parameters

- *--optim*: the optimizer name. It can be set to `kfac` (for KFAC optimizer), `kpsvd` (for KPSVD optimizer), `deflation` (for Deflation optimizer), `lanczos` (for Lanczos optimizer), `kfac_cor` (for KFAC corrected), `twolevel_kfac` (for two-level KFAC), `exactNG` (for exact natural gradient), `adam` (for ADAM) and `sgd` (for SGD)

- *--num_epochs*: number of epochs

- *--data* : the dataset. It can be `CURVES`, `MNIST` or `FACES` for the three MLP deep auto-encoder problems. for each data specified, a model architecture associated to it and implemented in `apps/mlp_cnn/models.py` is automatically set up. To train your own model with your own dataset, you need to define your model and data and call them in train_mlp.py` or train_cnn.py`

- *--batch_size*: batch size

- *--lr*: learning rate

- *--damping*: regularization parameter for the curvature matrix (only applies to second-order methods)

- *--clipping*: kl-clipping parameter (only apply to second-order method)

- *--momentum*: momentum parameter

- *--coarse_space*: the coarse space used in two-lvel KFAC. It can be `nicolaides`, `residual`, `tpselepedis` or `spectral`

- *--krylov*: wheter to use krylov or not to enrich the coarse space. It is 0 for no and 1 for yes. Only applies to two-level KFAC



## References

