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
$ git clone https://gitlab.ifpen.fr/supercalcul/natural-gradients.git
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

```sh
$ python apps/mlp_cnn/train_mlp.py --data CURVES --optim kfac --lr 1e-4 --damping 1e-4

```

All the default parameters of the functions `train_mlp.py` and `train_cnn.py` can be changed via `--` option. Below are different parameters

- *--optim*: the optimizer name. It can be set to `kfac` (for KFAC optimizer), `kpsvd` (for KPSVD optimizer), `deflation` (for Deflation optimizer), `lanczos` (for Lanczos optimizer), `kfac_cor` (for KFAC corrected), `twolevel_kfac` (for two-level KFAC), `exactNG` (for exact natural gradient), `adam` (for ADAM) and `sgd` (for SGD)

- *--num_epochs*: number of epochs

- *--data* : the dataset. It can be `CURVES`, `MNIST` or `FACES` for the three MLP deep auto-encoder problems. for each data specified, a model architecture associated to it and implemented in `apps/mlp_cnn/models.py` is automatically set up. To train your own model with your own dataset, you need to define your model and data and call them in train_mlp.py` or train_cnn.py`

- *--batch_size (int)*: batch size

- *--lr (int)*: learning rate

- *--damping (float)*: regularization parameter for the curvature matrix (only applies to second-order methods)

- *--clipping (float)*: kl-clipping parameter (only apply to second-order method)

- *--momentum (float)*: momentum parameter

- *--coarse_space (str)*: the coarse space used in two-lvel KFAC. It can be `nicolaides`, `residual`, `tpselepedis` or `spectral`

- *--krylov (str)*: wheter to use krylov or not to enrich the coarse space. It is 0 for no and 1 for yes. Only applies to two-level KFAC


### Deep convolutional auto-encoder

You can train a network containing **Transposed convolutional layer** with `KFAC` optimizer. Other natural gradient-based optimizers do not currently handle transposed convolutions. These methods should be extended to these types of convolutions in the very near future. Here, we provide an example of training convolutional auto-encoders that contain transposed convolutions. As in the case of the previous subsection, you just need to run the following command-line:

```sh
python apps/cnn_autoencoder/train.py --data --optim kfac
```

The parameter are the same as in the case of the previous subsection. But here the *--optim* argument can only be either `kfac`, `sgd` or `adam`. You can set the *--data*
argument to `MNIST` or `CIFAR10` to train a deep convolutional auto-encoder defined in `apps/cnn_autoencoder/train.py`. The model will correspond to the selected data. If you want to train your own model with your own data, you just need to create your model, load your dataset and call them in `apps/cnn_autoencoder/train.py` function.

### DCGANS

You can train DCGANs with the KFAC optimizer. You have the choice of training both the generator and the discriminator with KFAC, or training one of them with KFAC and the other with another optimizer (e.g. ADAM or SGD). Below is an example of training a DCGAN with the MNIST dataset, using the KFAC optimizer for both the generator and the discriminator.

```sh
$ cd apps/gans
$ python train.py --config configs/MNIST/kfac2.yml
```

Here you just need to privide as argument the path towards a **.yml** file containing the parameters of `train.py`. Examples of arguments are provided in `apps/gans/configs` folder. Below are what a *config.yml* file expects.

- *n_epochs (int)*: number of epoch
    
- *batch_size (int)*: batch size

- *device (str)*: the device for training the models. it can be set to `"cpu"` or `"cuda"`

- *z_dim (int)*: the dimension of noise vector used by the generator to generate fake images

- *gen_optim (str)*: the optimizer used to train the generator. It can be set to `"kfac"`, `"adam"` or `"sgda"`

- *crit_optim (str)*: the optimizer used to train the discriminator. It can be set to `"kfac"`, `"adam"` or `"sgda"`

- *gen_lr (float)*:  the learning rate of the genrator's optimizer

- *crit_lr (flaot)*: the learning rate of the discriminator's optimizer

- *clipping (float)*: the parameter for kl-clipping. Only applies when the optimizer is set to `"kfac"`

- *damping (float)*: the regularization parameter for the curvature matrix. Only applies when the optimizer is set to `"kfac"`

- *T_cov (int)*: the frequence of update of the curvature matrix. Only applies when the optimizer is set to `"kfac"`

- *T_inv (int)*: the frequence of computing the inverse of the curvature matrix. Only applies when the optimizer is set to `"kfac"`

- *momentum (float)*: the momentum parameter of sgd optimizer

- *c_lambda (float)*: the regularization parameter for gradient penality of the discriminator's loss

- *crit_repeats (int)*: the number of updates of the discriminator before an update of the generator

- *weight_decay (float)*: the deacay parameter for the optimizers

- *data_root (str)*: the path towards the datasets 




## Training on a supercomputer

Since most of clusters do not have access to the internet, it's difficult to configure python environments locally. Fortunately, conda-pack offers a solution for relocating conda environments to a new location. The full documentation is [here](https://conda.github.io/conda-pack/). 

First clone the repo, create your environement, install all required packages and conda-pack:

```sh
$ git clone https://gitlab.ifpen.fr/supercalcul/natural-gradients.git
$ cd natural-gradients
$ conda create --name my_env
$ conda activate my_env
$ conda install pip
$ pip install -r requirements.txt
$ conda install conda-pack
```

Then pack your environment:

```sh
$ conda-pack -f --ignore-missing-files --exclude lib/python3.1 --ignore-editable-packages

```
An archive `my_env.tar.gz` is ceated in the folder `natural-gradients`. Now you have to be copy the project on the supercomputer. For example, on `ener440`, you have to run

```sh
$ cd ..
$ scp -r natural-gradients <login>@ener440
```

After that, you have to go on the supercomputer, untar the file `my_env.tar.gz` contained in `natural-gradients` folder and source the environnement:

```sh
$ cd natural-gradients
$ mkdir my_env
$ tar -zxf my_env.tar.gz -C my_env
$ source my_env/bin/activate
```

Before using the package, make sure to install it before:

```sh
$ pip install -e .
```
You can now train your models on the supercomputer. 

## References

