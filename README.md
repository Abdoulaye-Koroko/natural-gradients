# Natural gradient-based optimization methods for deep learning

## Overview

This repository provides implementation of several natural gradient-based optimization methods for training deep neural networks. It include the following methods (see in references section for more on each method)

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


### MLP and CNN

You can train MLP or CNN networks with any optimizer using the `train_mlp.py` and `train_cnn.py` functions defined in the apps/mlp_cnn folder `apps/mlp_cnn`.

## References

