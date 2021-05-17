#!/usr/bin/env python
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import toml
import torch
from torch.utils.tensorboard import SummaryWriter

from config import load_config
from dataloaders.loader import ShapeBlurDataset, get_training_sample
from models.discriminator import Discriminator, TemporalDiscriminator
from models.encoder import EncoderCNN
from models.loss import FMOLoss, GANLoss, TemporalGANLoss, fmo_loss
from models.rendering import RenderingCNN
from utils import get_images


def main(args: Namespace) -> None:
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    encoder = EncoderCNN()
    rendering = RenderingCNN(config)
    loss_function = FMOLoss(config)

    if config.use_gan_loss:
        discriminator = Discriminator()
    if config.use_gan_timeconsistency:
        temp_disc = TemporalDiscriminator()

    l_temp_folder = args.run_folder / (
        datetime.now().isoformat() + "_defmotest"
    )
    if args.finetune_folder is not None:
        finetune_folder = args.finetune_folder.expanduser()
        encoder.load_state_dict(torch.load(finetune_folder / "encoder.pt"))
        rendering.load_state_dict(torch.load(finetune_folder / "rendering.pt"))
        if config.use_gan_loss:
            discriminator.load_state_dict(
                torch.load(finetune_folder / "discriminator.pt")
            )
        if config.use_gan_timeconsistency:
            temp_disc.load_state_dict(
                torch.load(finetune_folder / "temp_disc.pt")
            )

        if args.append_logs:
            l_temp_folder = finetune_folder

    encoder = torch.nn.DataParallel(encoder).to(device)
    rendering = torch.nn.DataParallel(rendering).to(device)

    if config.use_gan_loss:
        discriminator = torch.nn.DataParallel(discriminator).to(device)
        gan_loss_function = GANLoss(config)
    if config.use_gan_timeconsistency:
        temp_disc = torch.nn.DataParallel(temp_disc).to(device)
        temp_gan_function = TemporalGANLoss(config)

    if not l_temp_folder.exists():
        l_temp_folder.mkdir(parents=True)

    log_path = l_temp_folder / "training"
    if not log_path.exists():
        log_path.mkdir()

    with open(log_path / "config.toml", "w") as f:
        toml.dump(vars(config), f)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    rendering_params = sum(p.numel() for p in rendering.parameters())
    print(
        "Encoder params {:2f}M, rendering params {:2f}M".format(
            encoder_params / 1e6, rendering_params / 1e6
        ),
        end="",
    )

    if config.use_gan_loss:
        disc_params = sum(p.numel() for p in discriminator.parameters())
        print(
            ", discriminator params {:2f}M".format(disc_params / 1e6), end=""
        )
    if config.use_gan_timeconsistency:
        temp_disc_params = sum(p.numel() for p in temp_disc.parameters())
        print(
            ", temporal discriminator params {:2f}M".format(
                temp_disc_params / 1e6
            ),
            end="",
        )
    print("")

    dataset_folder = args.dataset_folder / "ShapeNetv2/ShapeBlur1000STL.hdf5"
    validation_folder = args.dataset_folder / "ShapeNetv2/ShapeBlur20STL.hdf5"

    training_set = ShapeBlurDataset(
        dataset_folder=dataset_folder,
        config=config,
        render_objs=config.render_objs_train,
        number_per_category=config.number_per_category,
        do_augment=True,
        use_latent_learning=config.use_latent_learning,
    )
    training_generator = torch.utils.data.DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_set = ShapeBlurDataset(
        dataset_folder=validation_folder,
        config=config,
        render_objs=config.render_objs_val,
        number_per_category=config.number_per_category_val,
        do_augment=True,
        use_latent_learning=False,
    )
    val_generator = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    vis_train_batch, _ = get_training_sample(
        dataset_folder, config, ["can"], min_obj=5, max_obj=5
    )
    vis_train_batch = vis_train_batch.unsqueeze(0).to(device)
    vis_val_batch, _ = get_training_sample(
        validation_folder, config, ["can"], min_obj=4, max_obj=4
    )
    vis_val_batch = vis_val_batch.unsqueeze(0).to(device)

    all_parameters = list(encoder.parameters()) + list(rendering.parameters())
    optimizer = torch.optim.Adam(all_parameters, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.sched_step_size, gamma=0.5
    )
    for _ in range(args.start_epoch):
        scheduler.step()
    writer = SummaryWriter(str(log_path), flush_secs=1)

    if config.use_gan_loss:
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=config.disc_lr
        )
        disc_scheduler = torch.optim.lr_scheduler.StepLR(
            disc_optimizer, step_size=config.sched_step_size, gamma=0.5
        )
        for _ in range(args.start_epoch):
            disc_scheduler.step()
    if config.use_gan_timeconsistency:
        temp_disc_optimizer = torch.optim.Adam(
            temp_disc.parameters(), lr=config.temp_disc_lr
        )
        temp_disc_scheduler = torch.optim.lr_scheduler.StepLR(
            temp_disc_optimizer, step_size=config.sched_step_size, gamma=0.5
        )
        for _ in range(args.start_epoch):
            temp_disc_scheduler.step()

    train_losses = []
    val_losses = []
    best_val_loss = 100.0
    for epoch in range(args.start_epoch, config.epochs):
        t0 = time.time()
        supervised_loss = []
        model_losses = []
        sharp_losses = []
        timecons_losses = []
        latent_losses = []
        if config.use_gan_loss:
            gen_losses = []
            disc_losses = []
        if config.use_gan_timeconsistency:
            temp_gen_losses = []
            temp_disc_losses = []
        joint_losses = []
        for it, (input_batch, times, hs_frames, times_left) in enumerate(
            training_generator
        ):
            encoder.train()
            rendering.train()
            if config.use_gan_loss:
                discriminator.train()
            if config.use_gan_timeconsistency:
                temp_disc.train()

            input_batch, times, hs_frames, times_left = (
                input_batch.to(device),
                times.to(device),
                hs_frames.to(device),
                times_left.to(device),
            )

            if config.use_latent_learning:
                latent = encoder(input_batch[:, :6])
                latent2 = encoder(input_batch[:, 6:])
            else:
                latent = encoder(input_batch)
                latent2 = []
            renders = rendering(latent, torch.cat((times, times_left), 1))

            sloss, mloss, shloss, tloss, lloss, jloss = loss_function(
                renders, hs_frames, input_batch[:, :6], (latent, latent2)
            )

            if config.use_gan_loss:
                for _ in range(config.disc_steps):
                    disc_optimizer.zero_grad()
                    disc_loss = gan_loss_function(
                        renders.detach(), hs_frames, discriminator
                    )[1]
                    disc_loss.mean().backward()
                    disc_optimizer.step()

                gen_loss, disc_loss = gan_loss_function(
                    renders, hs_frames, discriminator
                )
                jloss += config.gan_wt * gen_loss

            if config.use_gan_timeconsistency:
                for _ in range(config.temp_disc_steps):
                    temp_disc_optimizer.zero_grad()
                    temp_disc_loss = temp_gan_function(
                        renders.detach(), temp_disc
                    )[1]
                    temp_disc_loss.mean().backward()
                    temp_disc_optimizer.step()

                temp_gen_loss, temp_disc_loss = temp_gan_function(
                    renders, temp_disc
                )
                jloss += config.temp_gan_wt * temp_gen_loss

            supervised_loss.append(sloss.mean().item())
            model_losses.append(mloss.mean().item())
            sharp_losses.append(shloss.mean().item())
            timecons_losses.append(tloss.mean().item())
            latent_losses.append(lloss.mean().item())
            if config.use_gan_loss:
                gen_losses.append(gen_loss.mean().item())
                disc_losses.append(disc_loss.mean().item())
            if config.use_gan_timeconsistency:
                temp_gen_losses.append(temp_gen_loss.mean().item())
                temp_disc_losses.append(temp_disc_loss.mean().item())

            jloss = jloss.mean()
            joint_losses.append(jloss.item())

            global_step = epoch * len(training_generator) + it + 1
            if global_step % args.log_steps == 0:
                encoder.eval()
                rendering.eval()

                with torch.no_grad():
                    writer.add_scalar(
                        "Loss/train_joint", np.mean(joint_losses), global_step
                    )
                    print(
                        "Epoch {:4d}, it {:4d}".format(epoch + 1, it), end=" "
                    )

                    if config.use_supervised:
                        writer.add_scalar(
                            "Loss/train_supervised",
                            np.mean(supervised_loss),
                            global_step,
                        )
                        print(
                            f", loss {np.mean(supervised_loss):.3f}", end=" "
                        )
                    if config.use_selfsupervised_model:
                        writer.add_scalar(
                            "Loss/train_selfsupervised_model",
                            np.mean(model_losses),
                            global_step,
                        )
                        print(f", model {np.mean(model_losses):.3f}", end=" ")
                    if config.use_selfsupervised_sharp_mask:
                        writer.add_scalar(
                            "Loss/train_selfsupervised_sharpness",
                            np.mean(sharp_losses),
                            global_step,
                        )
                        print(f", sharp {np.mean(sharp_losses):.3f}", end=" ")
                    if config.use_selfsupervised_timeconsistency:
                        writer.add_scalar(
                            "Loss/train_selfsupervised_timeconsistency",
                            np.mean(timecons_losses),
                            global_step,
                        )
                        print(
                            f", time {np.mean(timecons_losses):.3f}", end=" "
                        )
                    if config.use_latent_learning:
                        writer.add_scalar(
                            "Loss/train_selfsupervised_latent",
                            np.mean(latent_losses),
                            global_step,
                        )
                        print(
                            f", latent {np.mean(latent_losses):.3f}", end=" "
                        )
                    if config.use_gan_loss:
                        writer.add_scalar(
                            "Loss/train_gan_generator",
                            np.mean(gen_losses),
                            global_step,
                        )
                        writer.add_scalar(
                            "Loss/train_gan_discriminator",
                            np.mean(disc_losses),
                            global_step,
                        )
                        print(f", gen {np.mean(gen_losses):.3f}", end=" ")
                        print(f", disc {np.mean(disc_losses):.3f}", end=" ")
                    if config.use_gan_timeconsistency:
                        writer.add_scalar(
                            "Loss/train_temp_gan_generator",
                            np.mean(temp_gen_losses),
                            global_step,
                        )
                        writer.add_scalar(
                            "Loss/train_temp_gan_discriminator",
                            np.mean(temp_disc_losses),
                            global_step,
                        )
                        print(
                            f", temp_gen {np.mean(temp_gen_losses):.3f}",
                            end=" ",
                        )
                        print(
                            f", temp_disc {np.mean(temp_disc_losses):.3f}",
                            end=" ",
                        )

                    print(f", joint {np.mean(joint_losses):.3f}")

                    writer.add_scalar(
                        "LR/value",
                        optimizer.param_groups[0]["lr"],
                        global_step,
                    )
                    writer.add_images(
                        "Vis Train Batch",
                        get_images(
                            encoder, rendering, device, vis_train_batch
                        )[0],
                        global_step,
                    )
                    writer.add_images(
                        "Vis Val Batch",
                        get_images(encoder, rendering, device, vis_val_batch)[
                            0
                        ],
                        global_step,
                    )

                    running_losses_min = []
                    running_losses_max = []
                    for it, (input_batch, times, hs_frames, _) in enumerate(
                        val_generator
                    ):
                        input_batch, times, hs_frames = (
                            input_batch.to(device),
                            times.to(device),
                            hs_frames.to(device),
                        )
                        latent = encoder(input_batch)
                        renders = rendering(latent, times)[:, :, :4]

                        val_loss1 = fmo_loss(renders, hs_frames)
                        val_loss2 = fmo_loss(
                            renders, torch.flip(hs_frames, [1])
                        )
                        losses = torch.cat(
                            (val_loss1.unsqueeze(0), val_loss2.unsqueeze(0)), 0
                        )
                        min_loss, _ = losses.min(0)
                        max_loss, _ = losses.max(0)
                        running_losses_min.append(min_loss.mean().item())
                        running_losses_max.append(max_loss.mean().item())
                    print(
                        "Step {:4d}, val it {:4d}, loss {}".format(
                            global_step, it, np.mean(running_losses_min)
                        )
                    )
                    val_losses.append(np.mean(running_losses_min))
                    if val_losses[-1] < best_val_loss and epoch >= 0:
                        torch.save(
                            encoder.module.state_dict(),
                            l_temp_folder / "encoder_best.pt",
                        )
                        torch.save(
                            rendering.module.state_dict(),
                            l_temp_folder / "rendering_best.pt",
                        )
                        if config.use_gan_loss:
                            torch.save(
                                discriminator.module.state_dict(),
                                l_temp_folder / "discriminator_best.pt",
                            )
                        if config.use_gan_timeconsistency:
                            torch.save(
                                temp_disc.module.state_dict(),
                                l_temp_folder / "temp_disc_best.pt",
                            )
                        best_val_loss = float(val_losses[-1])
                        print("    Saving best validation loss model!  ")

                    writer.add_scalar(
                        "Loss/val_min", val_losses[-1], global_step
                    )
                    writer.add_scalar(
                        "Loss/val_max",
                        np.mean(running_losses_max),
                        global_step,
                    )
                    concat = torch.cat(
                        (
                            renders[:, 0],
                            renders[:, -1],
                            hs_frames[:, 0],
                            hs_frames[:, -1],
                        ),
                        2,
                    )
                    writer.add_images(
                        "Val Batch",
                        concat[:, 3:] * (concat[:, :3] - 1) + 1,
                        global_step,
                    )
                    writer.flush()

            optimizer.zero_grad()
            jloss.backward()
            optimizer.step()

        train_losses.append(np.mean(supervised_loss))

        time_elapsed = (time.time() - t0) / 60
        print(
            f"Epoch {epoch+1:4d} took {time_elapsed:.2f} minutes, lr = "
            f"{optimizer.param_groups[0]['lr']}, av train loss "
            f"{train_losses[-1]:.5f}, val loss min {val_losses[-1]:.5f} max "
            f"{np.mean(running_losses_max):.5f}"
        )
        scheduler.step()
        if config.use_gan_loss:
            disc_scheduler.step()
        if config.use_gan_timeconsistency:
            temp_disc_scheduler.step()

    torch.cuda.empty_cache()
    torch.save(encoder.module.state_dict(), l_temp_folder / "encoder.pt")
    torch.save(rendering.module.state_dict(), l_temp_folder / "rendering.pt")
    if config.use_gan_loss:
        torch.save(
            discriminator.module.state_dict(),
            l_temp_folder / "discriminator.pt",
        )
    if config.use_gan_timeconsistency:
        torch.save(
            temp_disc.module.state_dict(), l_temp_folder / "temp_disc.pt"
        )
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train the DeFMO model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_folder", type=Path, help="Path to the dataset"
    )
    parser.add_argument(
        "run_folder",
        type=Path,
        help="Path where to dump logs and saved models for this run",
    )
    parser.add_argument(
        "--config", type=Path, help="Path to the TOML hyper-param config"
    )
    parser.add_argument(
        "--finetune-folder",
        type=Path,
        help="Path to the folder from where saved models should be loaded for "
        "fine-tuning",
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0, help="The epoch to start from"
    )
    parser.add_argument(
        "--append-logs",
        action="store_true",
        help="Whether to append to existing logs when fine-tuning",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="The number of workers for loading the input",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=200,
        help="step interval for logging summaries",
    )
    main(parser.parse_args())
