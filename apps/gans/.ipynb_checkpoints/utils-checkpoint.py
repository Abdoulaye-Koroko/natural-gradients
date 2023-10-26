import torch
from torchvision.utils import make_grid
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import Dataset, DataLoader
import zipfile
from PIL import Image
import os
import numpy as np

os.environ['TORCH_HOME'] = 'models'





from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ZipDataset(Dataset):
    
    def __init__(self, root_path="data/celebA.zip", transform = None):
        
        self.samples = []
        
        self.transform = transform
            
        with zipfile.ZipFile(root_path, mode="r") as archive:
            
            ids = archive.namelist()[1:]
            
            for id in ids:
            
                f = archive.open(id)

                try:
                    
                    sample = Image.open(f).convert('RGB')

                    self.samples +=[sample]

                except:

                    continue
            

    def __len__(self):
        
        return len(self.samples)

    def __getitem__(self, item):
        
        sample = self.samples[item]
        
        image = np.array(sample)
        
        if self.transform is not None:
            
            image = self.transform(image)  
        
        
        return image,0




def show_tensor_images(image_tensor, num_images=25,output_folder="",epoch=1):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.title(f"Epoch {epoch}")
    plt.axis('off')
    plt.show()
    plt.savefig(f'{output_folder}'+".jpg", format='jpg')
    plt.close()

def make_grad_hook():
    '''
    Function to keep track of gradients for visualization purposes, 
    which fills the grads list when using model.apply(grad_hook).
    '''
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook




def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient




def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = torch.mean((gradient_norm - 1)**2)
    #### END CODE HERE ####
    return penalty



def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    #### START CODE HERE ####
    gen_loss = -1. * torch.mean(crit_fake_pred)
    #### END CODE HERE ####
    return gen_loss



def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    #### START CODE HERE ####
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    #### END CODE HERE ####
    return crit_loss


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        

        
def plot_losses(step_bins,generator_losses,critic_losses,output_folder):
    
    num_examples = (len(generator_losses) // step_bins) * step_bins
    
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
        label="Generator Loss"
    )
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
        label="Critic Loss"
    )
    
    plt.xlabel('Iteration')
    
    plt.ylabel("Loss")
    
    plt.legend()
                
    plt.show()
    
    plt.savefig(f'{output_folder}'+".jpg", format='jpg',dpi=1200,transparent = True, bbox_inches = 'tight', pad_inches = 0)
    
    plt.close()
    
    
    

def delete_folder_contents(folder):
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def inception_score(images,device):
    
    inception = InceptionScore(feature=192,normalize=True).to(device)
    
    inception.update(images)
    
    mean,std = inception.compute()
    
    return mean


def plot_IS(step_bins,IS,output_folder):
                
    num_examples = (len(IS) // step_bins) * step_bins
    
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(IS[:num_examples]).view(-1, step_bins).mean(1),
        label="IS"
    )
    
    plt.legend()
    
    plt.xlabel('Iteration')
    
    plt.ylabel("IS")
                
    plt.show()
    
    plt.savefig(f'{output_folder}'+".jpg", format='jpg',dpi=1200,transparent = True, bbox_inches = 'tight', pad_inches = 0)
    
    plt.close() 
    
    
def fid(reals,fakes,device):
    
    fid = FrechetInceptionDistance(feature=192,normalize=True).to(device)
    
    fid.update(reals, real=True)
    
    fid.update(fakes, real=False) 
    
    FID = fid.compute()
    
    return FID


def plot_FID(step_bins,FID,output_folder):
    
    num_examples = (len(FID) // step_bins) * step_bins
                
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(FID[:num_examples]).view(-1, step_bins).mean(1),
        label="FID"
    )
    
    plt.legend()
    
    plt.xlabel('Iteration')
    
    plt.ylabel("FID")
                
    plt.show()
    
    plt.savefig(f'{output_folder}'+".jpg", format='jpg',dpi=1200,transparent = True, bbox_inches = 'tight', pad_inches = 0)
    
    plt.close() 
    
    
    