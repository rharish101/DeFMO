import argparse
import numpy as np
import random
import pdb
import os
import torch
import time
import toml

from utils import *
from vis import *
from main_settings import *

from models.encoder import *
from models.rendering import *
from models.discriminator import Discriminator, TemporalDiscriminator
from models.loss import *
from dataloaders.loader import *
from dataloaders.val_loaders import *
from helpers.torch_helpers import *
from torch.utils.tensorboard import SummaryWriter

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    encoder = EncoderCNN()
    rendering = RenderingCNN()
    loss_function = FMOLoss()

    if g_use_gan_loss:
        discriminator = Discriminator()
    if g_use_gan_timeconsistency:
        temp_disc = TemporalDiscriminator()

    l_temp_folder = g_temp_folder
    if g_finetune:
        encoder.load_state_dict(torch.load(os.path.join(g_load_temp_folder, 'encoder.pt')))
        rendering.load_state_dict(torch.load(os.path.join(g_load_temp_folder, 'rendering.pt')))
        if g_use_gan_loss:
            discriminator.load_state_dict(torch.load(os.path.join(g_load_temp_folder, 'discriminator.pt')))
        if g_use_gan_timeconsistency:
            temp_disc.load_state_dict(torch.load(os.path.join(g_load_temp_folder, 'temp_disc.pt')))
        if g_keep_logs:
            l_temp_folder = g_load_temp_folder

    encoder = nn.DataParallel(encoder).to(device)
    rendering = nn.DataParallel(rendering).to(device)

    if g_use_gan_loss:
        discriminator = nn.DataParallel(discriminator).to(device)
        gan_loss_function = GANLoss()
    if g_use_gan_timeconsistency:
        temp_disc = nn.DataParallel(temp_disc).to(device)
        temp_gan_function = TemporalGANLoss()

    if not os.path.exists(l_temp_folder):
        os.makedirs(l_temp_folder)

    log_path = os.path.join(l_temp_folder,'training')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, 'config.toml'), 'w') as f:
        config = {k: v for k, v in globals().items() if k.startswith('g_')}
        toml.dump(config, f)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    rendering_params = sum(p.numel() for p in rendering.parameters())
    encoder_grad = sum(int(p.requires_grad) for p in encoder.parameters())
    encoder_p = sum(1 for p in encoder.parameters())
    print('Encoder params {:2f}M, rendering params {:2f}M'.format(encoder_params/1e6,rendering_params/1e6), end='')

    if g_use_gan_loss:
        disc_params = sum(p.numel() for p in discriminator.parameters())
        print(', discriminator params {:2f}M'.format(disc_params/1e6), end='')
    if g_use_gan_timeconsistency:
        temp_disc_params = sum(p.numel() for p in temp_disc.parameters())
        print(', temporal discriminator params {:2f}M'.format(temp_disc_params/1e6), end='')
    print('')
    
    training_set = ShapeBlurDataset(dataset_folder=g_dataset_folder, render_objs = g_render_objs, number_per_category=g_number_per_category,do_augment=True,use_latent_learning=g_use_latent_learning)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=g_batch_size,shuffle=True,num_workers=g_num_workers,drop_last=True)
    val_set = ShapeBlurDataset(dataset_folder=g_validation_folder, render_objs = g_render_objs_val, number_per_category=g_number_per_category_val,do_augment=True,use_latent_learning=False)
    val_generator = torch.utils.data.DataLoader(val_set, batch_size=g_batch_size,shuffle=True,num_workers=g_num_workers,drop_last=True)

    vis_train_batch, _ = get_training_sample(["can"],min_obj=5,max_obj=5,dataset_folder=g_dataset_folder)
    vis_train_batch = vis_train_batch.unsqueeze(0).to(device)
    vis_val_batch, _ = get_training_sample(["can"],min_obj=4,max_obj=4,dataset_folder=g_validation_folder)
    vis_val_batch = vis_val_batch.unsqueeze(0).to(device)

    all_parameters = list(encoder.parameters()) + list(rendering.parameters())
    optimizer = torch.optim.Adam(all_parameters, lr=g_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=g_sched_step_size, gamma=0.5)
    for _ in range(g_start_epoch):
        scheduler.step()
    writer = SummaryWriter(log_path, flush_secs=1)

    if g_use_gan_loss:
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=g_disc_lr)
        disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=g_sched_step_size, gamma=0.5)
        for _ in range(g_start_epoch):
            disc_scheduler.step()
    if g_use_gan_timeconsistency:
        temp_disc_optimizer = torch.optim.Adam(temp_disc.parameters(), lr=g_temp_disc_lr)
        temp_disc_scheduler = torch.optim.lr_scheduler.StepLR(temp_disc_optimizer, step_size=g_sched_step_size, gamma=0.5)
        for _ in range(g_start_epoch):
            temp_disc_scheduler.step()

    train_losses = []
    val_losses = []
    best_val_loss = 100.0
    for epoch in range(g_start_epoch, g_epochs):
        encoder.train()
        rendering.train()
        if g_use_gan_loss:
            discriminator.train()
        if g_use_gan_timeconsistency:
            temp_disc.train()

        t0 = time.time()
        supervised_loss = []
        model_losses = []
        sharp_losses = []
        timecons_losses = []
        latent_losses = []
        if g_use_gan_loss:
            gen_losses = []
            disc_losses = []
        if g_use_gan_timeconsistency:
            temp_gen_losses = []
            temp_disc_losses = []
        joint_losses = []
        for it, (input_batch, times, hs_frames, times_left) in enumerate(training_generator):
            input_batch, times, hs_frames, times_left = input_batch.to(device), times.to(device), hs_frames.to(device), times_left.to(device)

            if g_use_latent_learning:
                latent = encoder(input_batch[:,:6])
                latent2 = encoder(input_batch[:,6:])
            else:
                latent = encoder(input_batch)
                latent2 = []
            renders = rendering(latent, torch.cat((times,times_left),1))

            sloss, mloss, shloss, tloss, lloss, jloss = loss_function(renders, hs_frames, input_batch[:,:6], (latent,latent2))

            if g_use_gan_loss:
                for _ in range(g_disc_steps):
                    disc_optimizer.zero_grad()
                    disc_loss = gan_loss_function(renders.detach(), hs_frames, discriminator)[1]
                    disc_loss.mean().backward()
                    disc_optimizer.step()

                gen_loss, disc_loss = gan_loss_function(renders, hs_frames, discriminator)
                jloss += g_gan_wt * gen_loss

            if g_use_gan_timeconsistency:
                for _ in range(g_temp_disc_steps):
                    temp_disc_optimizer.zero_grad()
                    temp_disc_loss = temp_gan_function(renders.detach(), temp_disc)[1]
                    temp_disc_loss.mean().backward()
                    temp_disc_optimizer.step()

                temp_gen_loss, temp_disc_loss = temp_gan_function(renders, temp_disc)
                jloss += g_temp_gan_wt * temp_gen_loss

            supervised_loss.append(sloss.mean().item())
            model_losses.append(mloss.mean().item())
            sharp_losses.append(shloss.mean().item())
            timecons_losses.append(tloss.mean().item())
            latent_losses.append(lloss.mean().item())
            if g_use_gan_loss:
                gen_losses.append(gen_loss.mean().item())
                disc_losses.append(disc_loss.mean().item())
            if g_use_gan_timeconsistency:
                temp_gen_losses.append(temp_gen_loss.mean().item())
                temp_disc_losses.append(temp_disc_loss.mean().item())

            jloss = jloss.mean()
            joint_losses.append(jloss.item())    

            if it % 10 == 0:
                global_step = epoch * len(training_generator) + it + 1
                writer.add_scalar('Loss/train_joint', np.mean(joint_losses), global_step)
                print("Epoch {:4d}, it {:4d}".format(epoch+1, it), end =" ")

                if g_use_supervised:
                    writer.add_scalar('Loss/train_supervised', np.mean(supervised_loss), global_step)
                    print(", loss {:.3f}".format(np.mean(supervised_loss)), end =" ")
                if g_use_selfsupervised_model:
                    writer.add_scalar('Loss/train_selfsupervised_model', np.mean(model_losses), global_step)
                    print(", model {:.3f}".format(np.mean(model_losses)), end =" ")
                if g_use_selfsupervised_sharp_mask:
                    writer.add_scalar('Loss/train_selfsupervised_sharpness', np.mean(sharp_losses), global_step)
                    print(", sharp {:.3f}".format(np.mean(sharp_losses)), end =" ")
                if g_use_selfsupervised_timeconsistency:
                    writer.add_scalar('Loss/train_selfsupervised_timeconsistency', np.mean(timecons_losses), global_step)
                    print(", time {:.3f}".format(np.mean(timecons_losses)), end =" ")
                if g_use_latent_learning:
                    writer.add_scalar('Loss/train_selfsupervised_latent', np.mean(latent_losses), global_step)
                    print(", latent {:.3f}".format(np.mean(latent_losses)), end =" ")
                if g_use_gan_loss:
                    writer.add_scalar('Loss/train_gan_generator', np.mean(gen_losses), global_step)
                    writer.add_scalar('Loss/train_gan_discriminator', np.mean(disc_losses), global_step)
                    print(", gen {:.3f}".format(np.mean(gen_losses)), end =" ")
                    print(", disc {:.3f}".format(np.mean(disc_losses)), end =" ")
                if g_use_gan_timeconsistency:
                    writer.add_scalar('Loss/train_temp_gan_generator', np.mean(temp_gen_losses), global_step)
                    writer.add_scalar('Loss/train_temp_gan_discriminator', np.mean(temp_disc_losses), global_step)
                    print(", temp_gen {:.3f}".format(np.mean(temp_gen_losses)), end =" ")
                    print(", temp_disc {:.3f}".format(np.mean(temp_disc_losses)), end =" ")

                print(", joint {:.3f}".format(np.mean(joint_losses)))

                writer.add_scalar('LR/value', optimizer.param_groups[0]['lr'], global_step)
                writer.add_images('Vis Train Batch', get_images(encoder, rendering, device, vis_train_batch)[0], global_step)
                writer.add_images('Vis Val Batch', get_images(encoder, rendering, device, vis_val_batch)[0], global_step)
                writer.flush()
            
            optimizer.zero_grad()
            jloss.backward()
            optimizer.step()

        train_losses.append(np.mean(supervised_loss))

        with torch.no_grad():
            encoder.eval()
            rendering.eval()
            if g_use_gan_loss:
                discriminator.eval()
            if g_use_gan_timeconsistency:
                temp_disc.eval()
            
            running_losses_min = []
            running_losses_max = []
            for it, (input_batch, times, hs_frames, _) in enumerate(val_generator):
                input_batch, times, hs_frames = input_batch.to(device), times.to(device), hs_frames.to(device)
                latent = encoder(input_batch)
                renders = rendering(latent, times)[:,:,:4]

                val_loss1 = fmo_loss(renders, hs_frames)
                val_loss2 = fmo_loss(renders, torch.flip(hs_frames,[1]))
                losses = torch.cat((val_loss1.unsqueeze(0),val_loss2.unsqueeze(0)),0)
                min_loss,_ = losses.min(0)
                max_loss,_ = losses.max(0)
                running_losses_min.append(min_loss.mean().item())
                running_losses_max.append(max_loss.mean().item())
            print("Epoch {:4d}, val it {:4d}, loss {}".format(epoch+1, it, np.mean(running_losses_min)))
            val_losses.append(np.mean(running_losses_min))
            if val_losses[-1] < best_val_loss and epoch >= 0:
                torch.save(encoder.module.state_dict(), os.path.join(l_temp_folder, 'encoder_best.pt'))
                torch.save(rendering.module.state_dict(), os.path.join(l_temp_folder, 'rendering_best.pt'))
                if g_use_gan_loss:
                    torch.save(discriminator.module.state_dict(), os.path.join(l_temp_folder, 'discriminator_best.pt'))
                if g_use_gan_timeconsistency:
                    torch.save(temp_disc.module.state_dict(), os.path.join(l_temp_folder, 'temp_disc_best.pt'))
                best_val_loss = val_losses[-1]
                print('    Saving best validation loss model!  ')
            
            global_step = (epoch + 1) * len(training_generator)
            writer.add_scalar('Loss/val_min', val_losses[-1], global_step)
            writer.add_scalar('Loss/val_max', np.mean(running_losses_max), global_step)
            concat = torch.cat((renders[:,0],renders[:,-1],hs_frames[:,0],hs_frames[:,-1]),2)
            writer.add_images('Val Batch', concat[:,3:]*(concat[:,:3]-1)+1, global_step)
            writer.flush()
            
        time_elapsed = (time.time() - t0)/60
        print('Epoch {:4d} took {:.2f} minutes, lr = {}, av train loss {:.5f}, val loss min {:.5f} max {:.5f}'.format(epoch+1, time_elapsed, optimizer.param_groups[0]['lr'], train_losses[-1], val_losses[-1], np.mean(running_losses_max)))
        scheduler.step()
        if g_use_gan_loss:
            disc_scheduler.step()
        if g_use_gan_timeconsistency:
            temp_disc_scheduler.step()
        
    # pdb.set_trace()
    torch.cuda.empty_cache()
    torch.save(encoder.module.state_dict(), os.path.join(l_temp_folder, 'encoder.pt'))
    torch.save(rendering.module.state_dict(), os.path.join(l_temp_folder, 'rendering.pt'))
    if g_use_gan_loss:
        torch.save(discriminator.module.state_dict(), os.path.join(l_temp_folder, 'discriminator.pt'))
    if g_use_gan_timeconsistency:
        torch.save(temp_disc.module.state_dict(), os.path.join(l_temp_folder, 'temp_disc.pt'))
    writer.close()

if __name__ == "__main__":
    main()
