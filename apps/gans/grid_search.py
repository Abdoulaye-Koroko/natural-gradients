import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import yaml
import argparse
import os 
from tqdm.auto import tqdm
import time
import itertools
import numpy as np

from apps.gans.utils import*
from apps.gans.models import*
from optimizers.kfac import KFAC



def train(config,lr,damp,clip):
    
    n_epochs = config['n_epochs']
    
    batch_size = config['batch_size']
    
    device = torch.device(config['device'])
    
    z_dim = config['z_dim']
    
    gen_optim = config['gen_optim']
    
    crit_optim = config['crit_optim']
    
    gen_lr = config['gen_lr']
    
    crit_lr = config['crit_lr']
    
    #display_step = config['display_step']
    
    c_lambda = config['c_lambda']
    
    crit_repeats = config['crit_repeats']
    
    data_root = config['data_root']
    
    data = config['data']
    
    weight_decay=config['weight_decay']
    
    
    if gen_optim=="kfac" and crit_optim=="kfac":
        
        output_folder = os.path.join("results",data,"params",gen_optim+"2")
    
    elif gen_optim=="kfac" and crit_optim!="kfac":
        
         output_folder = os.path.join("results",data,"params",gen_optim+"1")
            
    elif gen_optim!="kfac" and crit_optim=="kfac":
        
        output_folder = os.path.join("results",data,"params",crit_optim+"3")
        
    
    elif gen_optim=="sgd" and crit_optim=="sgd":
        
          output_folder = os.path.join("results",data,"params","sgda") 
    
    else:
         output_folder = os.path.join("results",data,"params",gen_optim)
         
        
    if not os.path.exists(output_folder):
        
        os.makedirs(output_folder,exist_ok=True) 
    
    train_dir = os.path.join(data_root,data)
        
    
    if data== "MNIST":
        
        transform = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

        
        training_set = datasets.MNIST(root=train_dir, train=True,
                                        download=False,transform=transform)

        dataloader =  torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True,drop_last=True)
        
        
        gen = Generator(z_dim).to(device)
    
        crit = Critic().to(device) 

        gen = gen.apply(weights_init)

        crit = crit.apply(weights_init)
        
        
    elif data=="CIFAR10":

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        training_set = datasets.CIFAR10(root=train_dir, train=True,
                                    download=False,transform=transform)

        dataloader =  torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                      shuffle=True,drop_last=True)


        ngf = 128

        ndf = 128

        nc = 3


        gen = ResNet32Generator(z_dim, nc, ngf, True).to(device)

        crit = ResNet32Discriminator(nc, 1, ndf, True).to(device) 

        gen.apply(weights_init)

        crit.apply(weights_init)

    
    
    if gen_optim == "adam":
        
        beta_1 = config['beta_1']
        
        beta_2 = config['beta_2']
        
        gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
        
        gen_precond = None
    
    elif gen_optim == "kfac":
        
        gen_opt = torch.optim.SGD(gen.parameters(),lr=lr,momentum=config['momentum'],nesterov=False,weight_decay=weight_decay)
            
        gen_precond = KFAC(gen,damp, pi=False, T_cov=config['T_cov'], T_inv=config['T_inv'],
                           alpha=0.95, constraint_norm=True,clipping=clip,batch_size=batch_size)
        
         
        
        gen_precond.update_stats = True
        
    
    elif gen_optim =="sgd":
        
        gen_opt = torch.optim.SGD(gen.parameters(),lr=lr,momentum=config['momentum'],nesterov=True,weight_decay=weight_decay)
        
        gen_precond = None 
             
    
    if crit_optim == "adam":
        
        beta_1 = config['beta_1']
        
        beta_2 = config['beta_2']
        
        crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))
        
        crit_precond = None
    
    elif crit_optim == "kfac":
        
        crit_opt = torch.optim.SGD(crit.parameters(),lr=lr,momentum=config['momentum'],nesterov=False,weight_decay=weight_decay)
            
        crit_precond = KFAC(crit,damp, pi=False, T_cov=config['T_cov'], T_inv=config['T_inv'],
                           alpha=0.95, constraint_norm=True,clipping=clip,batch_size=batch_size)
        
        crit_precond.update_stats = True
        
    elif crit_optim=="sgd":
        
        crit_opt = torch.optim.SGD(crit.parameters(),lr=lr,momentum=config['momentum'],nesterov=True,weight_decay=weight_decay)
        
        crit_precond = None
             
    
    
    
    cur_step = 0
    
    generator_losses = []
    
    critic_losses = []
    
    inception_scores = []
    
    fids = []
    
    display_step = len(dataloader)
    
    best_fid = 100000
    
    
    output_folder_fake = os.path.join(output_folder ,'fakes')
                
    output_folder_real = os.path.join(output_folder,'reals')
    
    #delete_folder_contents(output_folder_fake)
    
    #delete_folder_contents(output_folder_real)
    
    fixed_noise = get_noise(batch_size, z_dim, device=device)
    
    for epoch in range(n_epochs):
        
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            
            cur_batch_size = len(real)
            
            real = real.to(device)

            mean_iteration_critic_loss = 0
            
            for _ in range(crit_repeats):
                ### Update critic ###
                crit_opt.zero_grad()
                
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                
                fake = gen(fake_noise)
                
                crit_fake_pred = crit(fake.detach())
                
                crit_real_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                
                gradient = get_gradient(crit, real, fake.detach(), epsilon)
                
                gp = gradient_penalty(gradient)
                
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                if crit_precond is not None:
                    
                    crit_precond.step(update_params=True)
                    
                crit_opt.step()
                
            critic_losses += [mean_iteration_critic_loss]

            ### Update generator ###
            gen_opt.zero_grad()
            
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            
            fake_2 = gen(fake_noise_2)
            
            crit_fake_pred = crit(fake_2)

            gen_loss = get_gen_loss(crit_fake_pred)
            
            gen_loss.backward()

            # Update the weights
            if gen_precond is not None:
                
                gen_precond.step(update_params=True)
                
            gen_opt.step()

            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]
            
            
            #Scores
            if data=="MNIST":

                #IS = inception_score(fake.expand(-1,3,-1,-1),device)

                #inception_scores +=[IS]

                FID = fid(real.expand(-1,3,-1,-1),fake.expand(-1,3,-1,-1),device)

                fids +=[FID]

            else:

                #IS = inception_score(fake,device)

                #inception_scores +=[IS]

                FID = fid(real,fake,device)

                fids +=[FID] 



            ### Visualization code ###
            if cur_step % display_step == 0 and cur_step > 0:
                
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                
                #IS_mean = sum(inception_scores[-display_step:]) / display_step
                
                FID_mean = sum(fids[-display_step:]) / display_step

                if FID_mean<best_fid:
                    
                    best_fid = FID_mean
                
                print(f"Epoch {epoch}/{n_epochs}: Generator loss: {gen_mean}, critic loss: {crit_mean}, FID: {FID_mean}")


                #output_folder_fake_ =  os.path.join(output_folder_fake, str(epoch))
                
                #output_folder_real_ = os.path.join(output_folder_real, str(epoch))
                
                #show_tensor_images(image_tensor=gen(fixed_noise),output_folder=output_folder_fake_,epoch=epoch)
                
                #show_tensor_images(image_tensor=real, output_folder=output_folder_real_)
                
                #step_bins = 20
                
                #plot_losses(step_bins,generator_losses,critic_losses,os.path.join(output_folder,"losses"))
                
                #plot_IS(step_bins,inception_scores,os.path.join(output_folder,"IS"))
                
                #plot_FID(step_bins,fids,os.path.join(output_folder,"FID"))

            cur_step += 1
            
    
    
    return  best_fid,output_folder,batch_size

            
            
            
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Function arguments')

    parser.add_argument('--config', type=str, default ="configs/MNIST/adam.yml",
                    help='path to config')

    args = parser.parse_args() 
    
    with open(args.config, 'rb') as f:
        
        config = yaml.safe_load(f.read())  
    
    t1 = time.time()
    
    best_fid = 100000
    
    if config["crit_optim"]=="kfac" or config["gen_optim"]=="kfac":
    
        lrs = [1e-2]

        dampings = [1e-1,1e-2,1e-3,1e-4]

        clips = [0]

        list_params = [lrs,dampings,clips]

        for params in itertools.product(*list_params):

            print(f"Processing for params (lr,damp,kl): {params}")

            print(20*"+++++")

            print("\n")

            lr,damp,clip = params 

            #try:

            FID,output_folder,batch_size = train(config,lr,damp,clip)

            if FID<best_fid:

                best_params = {"lr":lr,"damping":damp,"clip":clip}

                best_fid = FID

            #except:
                #continue
    else: #Other optimizer
        
        lrs = [1e-1,1e-2,1e-3,1e-4]

        dampings = [0]

        kls = [0]

        list_params = [lrs,dampings,kls]

        for params in itertools.product(*list_params):

            print(f"Processing for params (lr,damp,kl): {params}")

            print(20*"+++++")

            print("\n")

            lr,damp,clip = params 

            try:

                FID,output_folder,batch_size = train(config,lr,damp,clip)

                if FID<best_fid:

                    best_params = {"lr":lr}

                    best_fid = FID

            except:
                
                continue
        
    print(f"Best params: {best_params}")
    
    print(f"Best fid: {best_fid}")
    
    npy_dir = output_folder 
    
    np.save(npy_dir+"_"+str(batch_size)+'_params.npy',best_params)
    
    t2 = time.time()
    
    print(f"Elapsed time: {(t2-t1)//60} min {(t2-t1)%60} s")