"""Hyper-param config handling."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import toml


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs."""

    number_per_category: int = 1000
    number_per_category_val: int = 20

    render_objs: Tuple[str, ...] = (
        "table",
        "jar",
        "skateboard",
        "bottle",
        "tower",
        "chair",
        "bookshelf",
        "camera",
        "laptop",
        "basket",
        "sofa",
        "knife",
        "can",
        "rifle",
        "train",
        "lamp",
        "trash bin",
        "mailbox",
        "watercraft",
        "motorbike",
        "dishwasher",
        "bench",
        "pistol",
        "rocket",
        "loudspeaker",
        "file cabinet",
        "bag",
        "cabinet",
        "bed",
        "birdhouse",
        "display",
        "piano",
        "earphone",
        "telephone",
        "stove",
        "microphone",
        "mug",
        "remote",
        "bathtub",
        "bowl",
        "keyboard",
        "guitar",
        "washer",
        "faucet",
        "printer",
        "cap",
        "clock",
        "helmet",
        "flowerpot",
        "microwaves",
    )
    render_objs_train: Tuple[str, ...] = render_objs
    render_objs_val: Tuple[str, ...] = render_objs

    epochs: int = 30
    batch_size: int = 7 * 3
    resolution_x: int = int(640 / 2)
    resolution_y: int = int(480 / 2)
    train_res_x: Optional[int] = None  # must be >= 32 and power of 2
    train_res_y: Optional[int] = None  # must be >= 32 and power of 2
    fmo_steps: int = 24
    fmo_train_steps: int = 2 * 12  # must be even
    use_median: bool = True
    normalize: bool = False

    sharp_mask_type: str = "entropy"
    timeconsistency_type: str = "ncc"  # oflow, ncc

    use_selfsupervised_model: bool = True
    use_selfsupervised_sharp_mask: bool = False
    use_selfsupervised_timeconsistency: bool = False
    use_supervised: bool = True
    use_latent_learning: bool = False
    use_gan_loss: bool = True
    use_nn_timeconsistency: bool = True

    lr: float = 1e-3
    disc_lr: float = 1e-5
    temp_disc_lr: float = 5e-5
    sched_step_size: int = 10

    gan_wt: float = 1.0
    temp_nn_wt: float = 0.05
    disc_steps: int = 1
    temp_disc_steps: int = 2

    seed: int = 0


def load_config(config_path: Optional[Path]) -> Config:
    """Load the hyper-param config at the given path.

    If the path doesn't exist, then an empty dict is returned.
    """
    if config_path is not None and config_path.exists():
        with open(config_path, "r") as f:
            args = toml.load(f)
    else:
        args = {}
    return Config(**args)
