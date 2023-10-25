import mat4py
import urllib
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch
from torchvision import datasets,transforms
from torch.utils.data import random_split
import sys
import pathlib

torch.manual_seed(0)


def maybe_download(SOURCE_URL, filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath




class DataSet(Dataset):
    def __init__(self, images):
        self._num_examples = images.shape[0]
        images = np.swapaxes(images, 2, 3)
        images = np.swapaxes(images, 1, 2)
        #images = images.reshape(images.shape[0],
                              #  images.shape[1] * images.shape[2] * images.shape[3])
        self.images = images.astype(np.float32)

        
        
    def __len__(self):
        
        return self._num_examples

    
    def __getitem__(self, item):
        
        image,label = torch.tensor(self.images[item]),torch.tensor(self.images[item])
        
        return image,label
        


def read_data_sets(name_dataset,batch_size):
   
    train_dir,name_dataset = name_dataset,os.path.split(name_dataset)[-1]
    #print(train_dir)
      
    if name_dataset == 'MNIST':
        print(f'Begin laoding {name_dataset} dataset ...')
        
        transform = transforms.Compose(
        [transforms.ToTensor()])


        training_set = datasets.MNIST(root=train_dir, train=True,
                                        download=True,transform=transform)
        
        test_abs = 50000
        train_subset, val_subset = random_split(
        training_set, [test_abs, len(training_set) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                          shuffle=True,drop_last=True)
        
        print(f"Num of training set: {len(train_subset)}")
        print(f"Num of Validation set: {len(val_subset)}")
        
        n_train = len(train_subset)
        n_val = len(val_subset)

        val_set = datasets.MNIST(root=train_dir, train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                         shuffle=False,drop_last=True)
        
        
        
    
    
    elif name_dataset =="F_MNIST":
        
        transform = transforms.Compose(
        [transforms.ToTensor()])
        
        training_set = datasets.FashionMNIST(root=train_dir, train=True,
                                        download=False,transform=transform)
        trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True,drop_last=True)
        
        val_set = datasets.FashionMNIST(root=train_dir, train=False,
                                       download=False, transform=transform)
        
        testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         shuffle=False,drop_last=True)
        
        n_train ,n_val = len(training_set),len(val_set)
        
        
    
    elif name_dataset == 'CIFAR10':
        print(f'Begin laoding {name_dataset} dataset ...')
        
        transform = transforms.Compose(
        [transforms.ToTensor()])


        training_set = datasets.CIFAR10(root=train_dir, train=True,
                                        download=True,transform=transform)
        trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True,drop_last=True)

        val_set = datasets.CIFAR10(root=train_dir, train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         shuffle=False,drop_last=True)
        
        print(f"Num of training set: {len(training_set)}")
        print(f"Num of Validation set: {len(val_set)}")
        
        n_train = len(training_set)
        n_val = len(val_set)
        
    elif name_dataset == 'CIFAR100':
        print(f'Begin laoding {name_dataset} dataset ...')
        
        transform = transforms.Compose(
        [transforms.ToTensor()])


        training_set = datasets.CIFAR100(root=train_dir, train=True,
                                        download=False,transform=transform)
        trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True,drop_last=True)

        val_set = datasets.CIFAR100(root=train_dir, train=False,
                                       download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         shuffle=False,drop_last=True)
        
        print(f"Num of training set: {len(training_set)}")
        print(f"Num of Validation set: {len(val_set)}")
        
        n_train = len(training_set)
        n_val = len(val_set)
    


        
    elif name_dataset == 'FACES':
        print(f'Begin laoding {name_dataset} dataset ...')
        SOURCE_URL = 'http://www.cs.toronto.edu/~jmartens/'
        TRAIN_IMAGES = 'newfaces_rot_single.mat'
        
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
    
        images_ = mat4py.loadmat(local_file)
        images_ = np.asarray(images_['newfaces_single'])
        images_ = np.transpose(images_)

        train_images = images_[:103500]
        test_images = images_[-41400:]

        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
        #train_labels = train_images
        #test_labels = test_images  
    
        training_set = DataSet(train_images)
        
        test_abs = int(len(training_set)*0.8)
        train_subset, val_subset = random_split(
        training_set, [test_abs, len(training_set) - test_abs])
        
        print(f"Num of training set: {len(train_subset)}")
        print(f"Num of Validation set: {len(val_subset)}")
        
        n_train = len(train_subset)
        n_val = len(val_subset)
        
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                          shuffle=True,drop_last=True)
        
        testloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                         shuffle=False,drop_last=True)


        
    elif name_dataset == 'CURVES':
        print(f'Begin laoding {name_dataset} dataset ...')
        
        SOURCE_URL = 'http://www.cs.toronto.edu/~jmartens/'
        TRAIN_IMAGES = 'digs3pts_1.mat'
        
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)

            
        images_ = mat4py.loadmat(local_file)
            
        train_images = np.asarray(images_['bdata'])
        test_images = np.asarray(images_['bdatatest'])
        
        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
        #train_labels = train_images
        #test_labels = test_images
        
        training_set = DataSet(train_images)
        
        test_abs = int(len(training_set)*0.8)
        train_subset, val_subset = random_split(
        training_set, [test_abs, len(training_set) - test_abs])
        
        print(f"Num of training set: {len(train_subset)}")
        print(f"Num of Validation set: {len(val_subset)}")
        
        n_train = len(train_subset)
        n_val = len(val_subset)
        
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                          shuffle=True,drop_last=True)
        
        testloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                         shuffle=False,drop_last=True)
        
    
    elif name_dataset=="D_MNIST":
        transform = transforms.Compose(
        [transforms.ToTensor()])
        
        training_set = datasets.MNIST(root=train_dir, train=True,
                                        download=False,transform=transform)
        trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True,drop_last=True)
        
        val_set = datasets.MNIST(root=train_dir, train=False,
                                       download=False, transform=transform)
        
        testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         shuffle=False,drop_last=True)
        
        n_train ,n_val = len(training_set),len(val_set)
    
    
    elif name_dataset == 'MNIST_RESTRICTED':
        print(f'Begin laoding {name_dataset} dataset ...')
        
        transform = transforms.Compose(
        [transforms.ToTensor()])

        
        training_set = datasets.MNIST(root=train_dir, train=True,
                                        download=False,transform=transform)
        
        indices = list(range(5000))
        
        train_subset = torch.utils.data.Subset(training_set,indices)
        
        
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                          shuffle=True,drop_last=True)
        
        
        val_set = datasets.MNIST(root=train_dir, train=False,
                                       download=False, transform=transform)
        
        val_subset = torch.utils.data.Subset(val_set,indices)
        
        testloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                         shuffle=False,drop_last=True) 
        
        
        print(f"Num of training set: {len(train_subset)}")
        print(f"Num of Validation set: {len(train_subset)}")
        
        n_train = len(train_subset)
        n_val = len(train_subset)

        
    
    
    
    
    elif name_dataset == 'CIFAR10_RESTRICTED':
        print(f'Begin laoding {name_dataset} dataset ...')
        
        transform = transforms.Compose(
        [transforms.ToTensor()])


        training_set = datasets.CIFAR10(root=train_dir, train=True,
                                        download=False,transform=transform)
        indices = list(range(5000))
        
        indices_val = list(range(1000))
        
        train_subset = torch.utils.data.Subset(training_set,indices)
        
        
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                          shuffle=True,drop_last=True)
        
        
        val_set = datasets.CIFAR10(root=train_dir, train=False,
                                       download=False, transform=transform)
        
        val_subset = torch.utils.data.Subset(val_set,indices_val)
        
        testloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                         shuffle=False,drop_last=True) 
        
        print(f"Num of training set: {len(train_subset)}")
        print(f"Num of Validation set: {len(val_subset)}")
        
        n_train = len(train_subset)
        n_val = len(train_subset)

        
     
    
    elif name_dataset == 'SVHN':
        print(f'Begin laoding {name_dataset} dataset ...')
        
        transform = transforms.Compose(
        [transforms.ToTensor()])


        training_set = datasets.SVHN(root=train_dir, split='train',
                                        download=False,transform=transform)
        trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True,drop_last=True)

        val_set = datasets.SVHN(root=train_dir, split='test',
                                       download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         shuffle=False,drop_last=True)
        #num_classes = training_set.classes
        
        #print(f"Num classes:{len(num_classes)}")
        print(f"Num of training set: {len(training_set)}")
        print(f"Num of Validation set: {len(val_set)}")
        
        n_train = len(training_set)
        n_val = len(val_set)
        
    
    
    else:
        print('error: Dataset not supported.')
        sys.exit()
    
        
    
    
    return n_train ,n_val , trainloader, testloader
