#!/usr/bin/env python
"""The main training script."""
from __future__ import annotations

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import toml
import torch
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Final

from config import Config, load_config
from dataloaders.loader import ShapeBlurDataset, get_training_sample
from models.discriminator import Discriminator, TemporalDiscriminator
from models.encoder import EncoderCNN
from models.loss import FMOLoss, GANLoss, TemporalNNLoss, fmo_loss
from models.rendering import RenderingCNN
from utils import get_images


@dataclass
class _Losses:
    """Dataclass for tracking losses."""

    supervised: torch.Tensor = 0.0
    model: torch.Tensor = 0.0
    sharp: torch.Tensor = 0.0
    timecons: torch.Tensor = 0.0
    latent: torch.Tensor = 0.0
    joint: torch.Tensor = 0.0
    gen: torch.Tensor = 0.0
    disc: torch.Tensor = 0.0
    temp_nn: torch.Tensor = 0.0

    def __truediv__(self, scalar: float) -> _Losses:
        return _Losses(
            **{key: value / scalar for key, value in vars(self).items()}
        )


class Trainer:
    """The class for training the model."""

    # Used when saving model weights
    ENC_PREFIX: Final = "encoder"
    RENDER_PREFIX: Final = "rendering"
    DISC_PREFIX: Final = "discriminator"
    TEMP_DISC_PREFIX: Final = "temp_disc"
    BEST_SUFFIX: Final = "_best"

    def __init__(
        self,
        config: Config,
        train_folder: Path,
        val_folder: Path,
        num_workers: int,
        save_folder: Path,
        load_folder: Optional[Path] = None,
        append_logs: bool = False,
    ):
        """Initialize everything involved in training.

        Args:
            config: The hyper-param config
            train_folder: The path to the training dataset
            val_folder: The path to the validation dataset
            num_workers: The number of processes for loading data
            save_folder: The path where to save logs and model weights
            load_folder: The path from where to load model weights from the
                previous run
            append_logs: Whether to overwrite logs from the previous run (only
                used if `load_folder` is not None)
        """
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._init_data(train_folder, val_folder, num_workers)
        self._init_logging(save_folder, load_folder, append_logs)
        self._init_models(load_folder)
        self._init_optimizers()

    def _init_data(
        self, train_folder: Path, val_folder: Path, num_workers: int
    ) -> None:
        train_dataset = ShapeBlurDataset(
            dataset_folder=train_folder.expanduser(),
            config=self.config,
            render_objs=self.config.render_objs_train,
            number_per_category=self.config.number_per_category,
            do_augment=True,
            use_latent_learning=self.config.use_latent_learning,
        )
        self.train_generator = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        val_dataset = ShapeBlurDataset(
            dataset_folder=val_folder.expanduser(),
            config=self.config,
            render_objs=self.config.render_objs_val,
            number_per_category=self.config.number_per_category_val,
            do_augment=True,
            use_latent_learning=False,
        )
        self.val_generator = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        vis_train_batch, _ = get_training_sample(
            train_folder, self.config, ["can"], min_obj=5, max_obj=5
        )
        self.vis_train_batch = vis_train_batch.unsqueeze(0).to(self.device)
        vis_val_batch, _ = get_training_sample(
            val_folder, self.config, ["can"], min_obj=4, max_obj=4
        )
        self.vis_val_batch = vis_val_batch.unsqueeze(0).to(self.device)

    def _init_logging(
        self, save_folder: Path, load_folder: Optional[Path], append_logs: bool
    ) -> None:
        self.save_folder = (
            save_folder.expanduser() / datetime.now().isoformat()
        )
        if load_folder is not None and append_logs:
            self.save_folder = load_folder.expanduser()

        if not self.save_folder.exists():
            self.save_folder.mkdir(parents=True)

        log_path = self.save_folder / "training"
        if not log_path.exists():
            log_path.mkdir()

        self.writer = SummaryWriter(str(log_path), flush_secs=1)
        with open(log_path / "config.toml", "w") as f:
            toml.dump(vars(self.config), f)

    def _init_models(self, load_folder: Optional[Path]) -> None:
        self.encoder = EncoderCNN()
        self.rendering = RenderingCNN(self.config)
        self.loss_fn = FMOLoss(self.config)

        if self.config.use_gan_loss:
            self.discriminator = Discriminator()
        if self.config.use_nn_timeconsistency:
            self.temp_disc = TemporalDiscriminator()

        if load_folder is not None:
            self.load_weights(load_folder)

        self.encoder = torch.nn.DataParallel(self.encoder).to(self.device)
        self.rendering = torch.nn.DataParallel(self.rendering).to(self.device)

        if self.config.use_gan_loss:
            self.discriminator = torch.nn.DataParallel(self.discriminator).to(
                self.device
            )
            self.gan_loss_fn = GANLoss(self.config)
        if self.config.use_nn_timeconsistency:
            self.temp_disc = torch.nn.DataParallel(self.temp_disc).to(
                self.device
            )
            self.temp_nn_fn = TemporalNNLoss(self.config)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        rendering_params = sum(p.numel() for p in self.rendering.parameters())
        print(
            f"Encoder params {encoder_params/1e6:2f}M, "
            f"rendering params {rendering_params/1e6:2f}M",
            end="",
        )

        if self.config.use_gan_loss:
            disc_params = sum(
                p.numel() for p in self.discriminator.parameters()
            )
            print(
                f", discriminator params {disc_params/1e6:2f}M",
                end="",
            )
        if self.config.use_nn_timeconsistency:
            temp_disc_params = sum(
                p.numel() for p in self.temp_disc.parameters()
            )
            print(
                f", temporal discriminator params {temp_disc_params/1e6:2f}M",
                end="",
            )
        print("")

    def _init_optimizers(self) -> None:
        model_params = list(self.encoder.parameters()) + list(
            self.rendering.parameters()
        )
        self.model_optim = torch.optim.Adam(model_params, lr=self.config.lr)
        self.model_sched = torch.optim.lr_scheduler.StepLR(
            self.model_optim, step_size=self.config.sched_step_size, gamma=0.5
        )

        if self.config.use_gan_loss:
            self.disc_optim = torch.optim.Adam(
                self.discriminator.parameters(), lr=self.config.disc_lr
            )
            self.disc_sched = torch.optim.lr_scheduler.StepLR(
                self.disc_optim,
                step_size=self.config.sched_step_size,
                gamma=0.5,
            )
        if self.config.use_nn_timeconsistency:
            self.temp_disc_optim = torch.optim.Adam(
                self.temp_disc.parameters(), lr=self.config.temp_disc_lr
            )
            self.temp_disc_sched = torch.optim.lr_scheduler.StepLR(
                self.temp_disc_optim,
                step_size=self.config.sched_step_size,
                gamma=0.5,
            )

    def load_weights(self, load_folder: Path) -> None:
        """Load weights from a previous checkpoint."""
        load_folder = load_folder.expanduser()

        self.encoder.load_state_dict(torch.load(load_folder / "encoder.pt"))
        self.rendering.load_state_dict(
            torch.load(load_folder / "rendering.pt")
        )

        if self.config.use_gan_loss:
            self.discriminator.load_state_dict(
                torch.load(load_folder / "discriminator.pt")
            )

        if self.config.use_nn_timeconsistency:
            self.temp_disc.load_state_dict(
                torch.load(load_folder / "temp_disc.pt")
            )

    def save_weights(self, save_best: bool = False) -> None:
        """Save weights to disk."""
        suffix = self.BEST_SUFFIX + ".pt" if save_best else ".pt"
        torch.save(
            self.encoder.module.state_dict(),
            self.save_folder / f"{self.ENC_PREFIX}{suffix}",
        )
        torch.save(
            self.rendering.module.state_dict(),
            self.save_folder / f"{self.RENDER_PREFIX}{suffix}",
        )
        if self.config.use_gan_loss:
            torch.save(
                self.discriminator.module.state_dict(),
                self.save_folder / f"{self.DISC_PREFIX}{suffix}",
            )
        if self.config.use_nn_timeconsistency:
            torch.save(
                self.temp_disc.module.state_dict(),
                self.save_folder / f"{self.TEMP_DISC_PREFIX}{suffix}",
            )

    def train(self, log_steps: int, start_epoch: int = 0) -> None:
        """Train the models."""
        # Advance the schedulers to match their state in the previous run
        for _ in range(start_epoch):
            self.model_sched.step()
            if self.config.use_gan_loss:
                self.disc_sched.step()
            if self.config.use_nn_timeconsistency:
                self.temp_disc_sched.step()

        best_val_loss = float("inf")
        running_losses = _Losses()
        global_step = start_epoch * len(self.train_generator) + 1

        for epoch in range(start_epoch, self.config.epochs):
            t0 = time.time()

            for it, inputs in enumerate(self.train_generator):
                running_losses = self._train_step(inputs, running_losses)
                global_step += 1

                if global_step % log_steps == 0:
                    curr_val_loss = self.save_logs(
                        global_step, running_losses / log_steps
                    )

                    # Reset loss tracking
                    running_losses = _Losses()

                    if curr_val_loss < best_val_loss:
                        self.save_weights(save_best=True)
                        best_val_loss = float(curr_val_loss)
                        print(
                            f"Step {global_step}: Saving best validation loss "
                            "model!"
                        )

            time_elapsed = (time.time() - t0) / 60
            print(f"Epoch {epoch+1:4d} took {time_elapsed:.2f} minutes")
            self.model_sched.step()
            if self.config.use_gan_loss:
                self.disc_sched.step()
            if self.config.use_nn_timeconsistency:
                self.temp_disc_sched.step()

        torch.cuda.empty_cache()
        self.save_weights()
        self.writer.close()

    def _train_step(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        running_losses: _Losses,
    ) -> _Losses:
        self.encoder.train()
        self.rendering.train()
        if self.config.use_gan_loss:
            self.discriminator.train()
        if self.config.use_nn_timeconsistency:
            self.temp_disc.train()

        input_batch = inputs[0].to(self.device)
        times = inputs[1].to(self.device)
        hs_frames = inputs[2].to(self.device)
        times_left = inputs[3].to(self.device)

        if self.config.use_latent_learning:
            latent = self.encoder(input_batch[:, :6])
            latent2 = self.encoder(input_batch[:, 6:])
        else:
            latent = self.encoder(input_batch)
            latent2 = []

        renders = self.rendering(latent, torch.cat((times, times_left), 1))

        sloss, mloss, shloss, tloss, lloss, jloss = self.loss_fn(
            renders, hs_frames, input_batch[:, :6], (latent, latent2)
        )
        running_losses.supervised += sloss.mean().item()
        running_losses.model += mloss.mean().item()
        running_losses.sharp += shloss.mean().item()
        running_losses.timecons += tloss.mean().item()
        running_losses.latent += lloss.mean().item()

        if self.config.use_gan_loss:
            for _ in range(self.config.disc_steps):
                self.disc_optim.zero_grad()
                disc_loss = self.gan_loss_fn(
                    renders.detach(), hs_frames, self.discriminator
                )[1]
                disc_loss.mean().backward()
                self.disc_optim.step()

            gen_loss, disc_loss = self.gan_loss_fn(
                renders, hs_frames, self.discriminator
            )
            running_losses.gen += gen_loss.mean().item()
            running_losses.disc += disc_loss.mean().item()
            jloss += self.config.gan_wt * gen_loss

        if self.config.use_nn_timeconsistency:
            for _ in range(self.config.temp_disc_steps):
                self.temp_disc_optim.zero_grad()
                temp_nn_loss = self.temp_nn_fn(
                    renders.detach(), self.temp_disc
                )
                temp_nn_loss.mean().backward()
                self.temp_disc_optim.step()

            temp_nn_loss = self.temp_nn_fn(renders, self.temp_disc)
            running_losses.temp_nn += temp_nn_loss.mean().item()
            jloss += self.config.temp_nn_wt * temp_nn_loss

        jloss = jloss.mean()
        running_losses.joint += jloss.item()

        self.model_optim.zero_grad()
        jloss.backward()
        self.model_optim.step()

        return running_losses

    def save_logs(
        self,
        global_step: int,
        loss: _Losses,
    ) -> float:
        """Save logs to disk."""
        self.encoder.eval()
        self.rendering.eval()

        with torch.no_grad():
            self.writer.add_scalar("Loss/train_joint", loss.joint, global_step)

            if self.config.use_supervised:
                self.writer.add_scalar(
                    "Loss/train_supervised", loss.supervised, global_step
                )
            if self.config.use_selfsupervised_model:
                self.writer.add_scalar(
                    "Loss/train_selfsupervised_model", loss.model, global_step
                )
            if self.config.use_selfsupervised_sharp_mask:
                self.writer.add_scalar(
                    "Loss/train_selfsupervised_sharpness",
                    loss.sharp,
                    global_step,
                )
            if self.config.use_selfsupervised_timeconsistency:
                self.writer.add_scalar(
                    "Loss/train_selfsupervised_timeconsistency",
                    loss.timecons,
                    global_step,
                )
            if self.config.use_latent_learning:
                self.writer.add_scalar(
                    "Loss/train_selfsupervised_latent",
                    loss.latent,
                    global_step,
                )
            if self.config.use_gan_loss:
                self.writer.add_scalar(
                    "Loss/train_gan_generator", loss.gen, global_step
                )
                self.writer.add_scalar(
                    "Loss/train_gan_discriminator", loss.disc, global_step
                )
            if self.config.use_nn_timeconsistency:
                self.writer.add_scalar(
                    "Loss/train_temp_nn", loss.temp_nn, global_step
                )

            self.writer.add_scalar(
                "LR/value", self.model_optim.param_groups[0]["lr"], global_step
            )
            self.writer.add_images(
                "Vis Train Batch",
                get_images(
                    self.encoder,
                    self.rendering,
                    self.device,
                    self.vis_train_batch,
                )[0],
                global_step,
            )
            self.writer.add_images(
                "Vis Val Batch",
                get_images(
                    self.encoder,
                    self.rendering,
                    self.device,
                    self.vis_val_batch,
                )[0],
                global_step,
            )

            val_min, val_max, val_batch = self._get_val_loss()
            self.writer.add_scalar("Loss/val_min", val_min, global_step)
            self.writer.add_scalar("Loss/val_max", val_max, global_step)
            self.writer.add_images("Val Batch", val_batch, global_step)

            self.writer.flush()
            return val_min

    def _get_val_loss(self) -> Tuple[float, float, torch.Tensor]:
        min_loss = 0.0
        max_loss = 0.0

        for it, (input_batch, times, hs_frames, _) in enumerate(
            self.val_generator
        ):
            input_batch, times, hs_frames = (
                input_batch.to(self.device),
                times.to(self.device),
                hs_frames.to(self.device),
            )
            latent = self.encoder(input_batch)
            renders = self.rendering(latent, times)[:, :, :4]

            val_loss1 = fmo_loss(renders, hs_frames)
            val_loss2 = fmo_loss(renders, torch.flip(hs_frames, [1]))
            losses = torch.cat(
                (val_loss1.unsqueeze(0), val_loss2.unsqueeze(0)), 0
            )
            min_loss += losses.min(0)[0].mean().item()
            max_loss += losses.max(0)[0].mean().item()

        min_loss /= len(self.val_generator)
        max_loss /= len(self.val_generator)

        concat = torch.cat(
            (renders[:, 0], renders[:, -1], hs_frames[:, 0], hs_frames[:, -1]),
            2,
        )
        val_batch = concat[:, 3:] * (concat[:, :3] - 1) + 1

        return min_loss, max_loss, val_batch


def main(args: Namespace) -> None:
    """Run the main function."""
    config = load_config(args.config)
    train_folder = args.dataset_folder / "ShapeNetv2/ShapeBlur1000STL.hdf5"
    val_folder = args.dataset_folder / "ShapeNetv2/ShapeBlur20STL.hdf5"

    trainer = Trainer(
        config,
        train_folder,
        val_folder,
        args.num_workers,
        args.run_folder,
        load_folder=args.finetune_folder,
        append_logs=args.append_logs,
    )
    trainer.train(log_steps=args.log_steps, start_epoch=args.start_epoch)


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
