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
$ conda create --name my_env
$ conda activate my_env
$ conda install pip
$ pip install -e .
$ pip install -r requirements.txt
```

## Usage

You can use the optimizers to train several types of deep neural networks. Below are the different types of compatible layers.
- Multi-layer perceptrons (MLP) layers : all optimizer

- Convolutional layers: all optimizer

- Transposed convolutional layers: KFAC


### General use case

To use one of the proposed natural-gradient methods, you can write your training function as follows.

#### Using the true Fisher
Since the Fisher Information Matrix (FIM) is defined as an expectation with respect to model's joint distribution, we need to estimate it with Monte-Carlo method. So it's necessary to use inputs $x$'s from the training data and targets y's sampled from the model's conditional distribution. The model's conditional distribution is defined by the loss function used to train the network. For most pratical loss functions, the model's conditional distribution is straithfoward to define. For example, when the loss function is the *mean square error* or the L2 norm, the model distribution is the **multivariate normal distribution**. When It 's the *binary-cross-entropy* loss function, the model distribution can be taken as the **Bernoulli distribution**. And finally, when it's the *cross-entropy* loss function, the model's conditional distribution is defined as the **multinomial distribution**.

Here is the code snapset of training a model with natural gradient-based method based on an estimation of the true Fisher.

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

#Training loop
for epoch in range(num_epochs):
    
    model.train()
    
    for iter,batch in enumerate(trainloader):
        
        optimizer.zero_grad()
        
        inputs,labels = batch
        
        inputs,labels = inputs.to(device),labels.to(device)
        
        with torch.set_grad_enabled(True):
            
            outputs = model(inputs)
            
            loss = criterion(outputs,labels)


        preconditioner.update_stats = True
        
        index = np.random.randint(low=0, high=batch_size, size=fisher_batch_size) # fisher_batch_size is the size of the batch used to compute the curvature matrix
        
        outputs_fisher = model(inputs[index])
        
        with torch.no_grad():
            
            sampled_y = model_conditional_distribution(outputs_fisher) # Sample from the model's conditional distribution
            
            # model_conditional_distribution should be defined according to the loss function. For example if the loss function is 
            
            # - the cross-entropy loss function, then sampled_y can be defined as follows:
            
                # sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs_fisher.cpu().data, dim=1),1).squeeze().to(device)
            
            # - the binary cross-entropy loss function, sampled_y is defined as follows :
            
                # sampled_y = torch.bernoulli(outputs_fisher)
                
            # - the mean square error, then sampled_y can be defined by:
            
                # sampled_y = torch.normal(mean=outputs_fisher)
            
        loss_sample = criterion(outputs_fisher,sampled_y)
        
        loss_sample.backward(retain_graph=True)
        
        preconditioner.step(update_params=False) 
        
        optimizer.zero_grad()
        
        preconditioner.update_stats = False

        loss.backward()

        preconditioner.step(update_params=True) # Preconditionnes the gradients with the computed Fisher   

        optimizer.step()
```

#### Using the empirical Fisher

Sometimes, it can be difficult to sample targets y's from the model's conditional distribution. This is mainly due to a complex loss function or a huge cost of sampling from the model's conditional distribution in muti-dimensional case.  In such a situation, one can use targets y's from the training data to estimate the Fisher. In that case, the curvature is said to be an estimation of the empirical Fisher. 

Below is a way to use a natural gradient-based optimization based on an estimation of the empirical Fisher.

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
for epoch in range(num_epochs):
    
    model.train()
    
    for iter,batch in enumerate(trainloader):

        optimizer.zero_grad()

        inputs,labels = batch

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

You can train MLP or CNN networks with any optimizer using the `train_mlp.py` aor `train_cnn.py` functions defined in the apps/mlp_cnn folder `apps/mlp_cnn`.

For example, to train the MLP deep auto-encoder with the CURVES dataset with KFAC optimizer, you just need to run the following command:

```sh
$ python apps/mlp_cnn/train_mlp.py --data CURVES --data_path ./data/ --optim kfac --lr 1e-4 --damping 1e-4 --num_epochs 10 --result_name kfac

```

All the default parameters of the functions `train_mlp.py` and `train_cnn.py` can be changed via `--` option. Below are different parameters

- *--optim*: the optimizer name. It can be set to `kfac` (for KFAC optimizer), `kpsvd` (for KPSVD optimizer), `deflation` (for Deflation optimizer), `lanczos` (for Lanczos optimizer), `kfac_cor` (for KFAC corrected), `twolevel_kfac` (for two-level KFAC), `exactNG` (for exact natural gradient), `adam` (for ADAM) and `sgd` (for SGD)

- *--num_epochs (int)*: number of epochs

- *--data (str)* : the dataset. It can be `CURVES`, `MNIST` or `FACES` for the three MLP deep auto-encoder problems. for each data specified, a model architecture associated to it and implemented in `apps/mlp_cnn/models.py` is automatically set up. To train your own model with your own dataset, you need to define your model and data and call them in train_mlp.py` or train_cnn.py`

- *--batch_size (int)*: batch size

- *--lr (int)*: learning rate

- *--damping (float)*: regularization parameter for the curvature matrix (only applies to second-order methods)

- *--clipping (float)*: kl-clipping parameter (only apply to second-order method)

- *--momentum (float)*: momentum parameter

- *--coarse_space (str)*: the coarse space used in two-lvel KFAC. It can be `nicolaides`, `residual`, `tpselepedis` or `spectral`

- *--krylov (str)*: wheter to use krylov or not to enrich the coarse space. It is 0 for no and 1 for yes. Only applies to two-level KFAC

- *data_path (str)*: the path towards the folder containing the dataset

- *result_name (str)*: the name under which the results are saved.


### Deep convolutional auto-encoder

You can train a network containing **Transposed convolutional layer** with `KFAC` optimizer. Other natural gradient-based optimizers do not currently handle transposed convolutions. These methods should be extended to these types of convolutions in the very near future. Here, we provide an example of training convolutional auto-encoders that contain transposed convolutions. As in the case of the previous subsection, you just need to run the following command-line:

```sh
python apps/cnn_autoencoder/train.py --optim kfac --data MNIST --data_path ./data/ --lr 1e-4 --damping 1e-4 --num_epochs 1 --result_name kfac
```

The parameter are the same as in the case of the previous subsection. But here the *--optim* argument can only be either `kfac`, `sgd` or `adam`. You can set the *--data*
argument to `MNIST` or `CIFAR10` to train a deep convolutional auto-encoder defined in `apps/cnn_autoencoder/train.py`. The model will correspond to the selected data. If you want to train your own model with your own data, you just need to create your model, load your dataset and call them in `apps/cnn_autoencoder/train.py` function.

### DCGANS

You can train DCGANs with the KFAC optimizer. You have the choice of training both the generator and the discriminator with KFAC, or training one of them with KFAC and the other with another optimizer (e.g. ADAM or SGD). Below is an example of training a DCGAN with the MNIST dataset, using the KFAC optimizer for both the generator and the discriminator.

```sh
$ python apps/gans/train.py --config apps/gans/configs/MNIST/kfac2.yml --result_name kfac2
```

Here you just need to privide as arguments the path towards a **.yml** file containing the parameters of `train.py` and *--result_name* the name under which the results will be saved. Examples of arguments are provided in `apps/gans/configs` folder. Below are what a *config.yml* file expects.

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

Bellow is an example of a job to simultaneously train 4 models with 4 differents GPUs on **Topaze** supercomputer.

```
#!/bin/bash
#MSUB -r my_job # JOB NAME
#MSUB -N 1 #Number of nodes
#MSUB -n 4
#MSUB -c 32
#MSUB -T 259200 # Wall time in seconds
#MSUB -o outpout.out # Output file
#MSUB -e error.err # Error file
#MSUB -q a100
#MSUB -m scratch
#MSUB -Q long

set -x

srun -n 1 -c 32 '--gpus=1' --exclusive python3 apps/mlp_cnn/train_mlp.py --data CURVES --data_path ./data/  --optim kfac --lr 1e-4 --damping 1e-4 --result_name kfac > output_kfac.out &

srun -n 1 -c 32 '--gpus=1' --exclusive python3 apps/mlp_cnn/train_mlp.py --data CURVES --data_path ./data/  --optim kpsvd --lr 1e-4 --damping 1e-4 --result_name kpsvd > output_kpsvd.out &

srun -n 1 -c 32 '--gpus=1' --exclusive python3 apps/mlp_cnn/train_mlp.py --data CURVES --data_path ./data/  --optim deflation --lr 1e-4 --damping 1e-4 --result_name deflation > output_deflation.out &

srun -n 1 -c 32 '--gpus=1' --exclusive python3 apps/mlp_cnn/train_mlp.py --data CURVES --data_path ./data/   --optim kfac_cor --lr 1e-4 --damping 1e-4 --result_name kfac_cor> output_kfac_cor.out &

wait
```

Let's say you save the above job un the name `my_job.sh`. You can submit your job on the supercomputer with the following command-line:

```sh
$ ccc_msub my_job.sh
```

## References
```
@ARTICLE{Amari1998,
  author = {Amari, Shun-Ichi},
  title = {Natural Gradient Works Efficiently in Learning},
  journal = {Neur. Comput.},
  year = {1998},
  volume = {10},
  pages = {251--276},
  number = {2},
  doi = {10.1162/089976698300017746},
}
@inproceedings{MartensGrosse2015,
  author = {Martens, James and Grosse, Roger},
  title = {Optimizing neural networks with {K}ronecker-factored approximate
	curvature},
  editor = {Bach, Francis and Blei, David},
  booktitle = {32nd International Conference on Machine Learning},
  year = {2015},
  month = {6--11 Jul},
  series = {Proceedings in Machine Learning Research},
  volume = {37},
  pages = {2408--2417},
  address = {Lille, France},
  url = {http://proceedings.mlr.press/v37/martens15.html}
}

@INPROCEEDINGS{GrosseMartens2016,
  author = {Grosse, Roger and Martens, James},
  title = {A {K}ronecker-factored approximate {F}isher matrix for convolution
	layers},
  booktitle = {33rd International Conference on Machine Learning},
  editor = {Balcan, Maria Florina and Weinberger, Kilian Q.},
  year = {2016},
  month = {19--24 Jun},
  series = {Proceedings of Machine Learning Research},
  volume = {48},
  pages = {573--582},
  address = {New York},
  url = {http://proceedings.mlr.press/v48/grosse16.html}
}

@ARTICLE{refId0,
	author = {Koroko, Abdoulaye and Anciaux-Sedrakian, Ani and Gharbia, Ibtihel Ben and Gar\`es, Val\'erie and Haddou, Mounir and Tran, Quang Huy},
	title = {Efficient approximations of the {F}isher matrix in neural networks using {K}ronecker product singular value decomposition},
	DOI = {10.1051/proc/202373218},
	journal = {ESAIM: ProcS},
	year = {2023},
	volume = {73},
	pages = {218--237},
}

@misc{Koroko2023analysis,
  title = {Analysis and Comparison of Two-Level {KFAC} Methods for Training Deep Neural Networks}, 
  author = {Koroko, Abdoulaye and Anciaux-Sedrakian, Ani and Ben Gharbia, Ibtihel
            and Garès, Valérie and Haddou, Mounir and Tran, Quang Huy},
  year = {2023},
  eprint = {2303.18083},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
@INPROCEEDINGS{twolevels,
  author = {Tselepidis, Nikolaos and Kohler, Jonas and Orvieto, Antonio},
  title = {Two-Level {K-FAC} Preconditioning for Deep Learning},
  booktitle = {12th Annual Workshop on Optimization for Machine Learning},
  address = {virtual},
  year = {2020},
  url = {https://www.opt-ml.org/papers/2020/paper_63.pdf}
}
@inproceedings{Benzing2022GradientDO,
  title = {Gradient Descent on Neurons and its Link to Approximate Second-Order Optimization},
  author = {Benzing, Frederik},
  booktitle = {39th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {162},
  year = {2022},
  month = {17--23 Jul},
  address = {Baltimore, Maryland},
  url = {https://proceedings.mlr.press/v162/benzing22a.html}
}

@ARTICLE{VanLoan2000,
  author = {van Loan, Charles F.},
  title = {The ubiquitous {K}ronecker product},
  journal = {J. Comput. Appl. Math.},
  year = {2000},
  volume = {123},
  pages = {85--100},
  number = {1},
  doi = {10.1016/S0377-0427(00)00393-9}
}

```

