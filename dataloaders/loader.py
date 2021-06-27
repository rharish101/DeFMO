import random
from pathlib import Path
from typing import Callable, List, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import Config


class ShapeBlurDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_folder: Path,
        config: Config,
        render_objs: Sequence[str],
        number_per_category: int,
        do_augment: bool = False,
        use_latent_learning: bool = False,
    ):
        self.timestamps = torch.linspace(0, 1, config.fmo_steps)
        self.dataset_folder = dataset_folder
        self.config = config
        self.render_objs = render_objs
        self.number_per_category = number_per_category
        self.do_augment = do_augment
        self.use_latent_learning = use_latent_learning
        self.crop_fn = transforms.RandomCrop(
            [self.config.train_res_x, self.config.train_res_y]
        )

    def __len__(self) -> int:
        return len(self.render_objs) * self.number_per_category

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        objname = int(index / self.number_per_category)
        objid = (index % self.number_per_category) + 1
        inputs, gt_paths = get_training_sample(
            dataset_folder=self.dataset_folder,
            config=self.config,
            render_objs=[self.render_objs[objname]],
            min_obj=objid,
            max_obj=objid,
            use_latent_learning=self.use_latent_learning,
        )

        perm = torch.randperm(int(self.config.fmo_steps / 2))
        inds = perm[: int(self.config.fmo_train_steps / 2)]
        inds, _ = inds.sort()

        inds = torch.cat(
            (inds, (self.config.fmo_steps - 1) - torch.flip(inds, [0])), 0
        )
        times = self.timestamps[inds]

        inds_left = perm[int(self.config.fmo_train_steps / 2) :]
        inds_left = torch.cat(
            (
                inds_left,
                (self.config.fmo_steps - 1) - torch.flip(inds_left, [0]),
            ),
            0,
        )
        times_left = self.timestamps[inds_left]

        if isinstance(gt_paths, list):
            hs_frames = []
            for ind in inds:
                gt_batch = get_gt_sample(gt_paths, ind)
                hs_frames.append(gt_batch)
            hs_frames = torch.stack(hs_frames, 0).contiguous()
        else:
            hs_frames = gt_paths[inds]

        if self.do_augment:
            if random.random() > 0.5:
                inputs = torch.flip(inputs, [-1])
                hs_frames = torch.flip(hs_frames, [-1])
            if random.random() > 0.5:
                inputs = torch.flip(inputs, [-2])
                hs_frames = torch.flip(hs_frames, [-2])

        if (
            self.config.train_res_x is not None
            and self.config.train_res_y is not None
        ):
            inputs = self.crop_fn(inputs)
            hs_frames = self.crop_fn(hs_frames)

        return inputs, times, hs_frames, times_left


def get_transform(
    normalize: bool,
) -> Callable[[Union[Image.Image, np.ndarray]], torch.Tensor]:
    if normalize:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.ToTensor()


def get_training_sample(
    dataset_folder: Path,
    config: Config,
    render_objs: Sequence[str],
    max_obj: int,
    min_obj: int = 1,
    use_latent_learning: bool = False,
) -> Tuple[torch.Tensor, List[str]]:
    use_hdf5 = "hdf5" in dataset_folder.name
    if use_hdf5:
        h5_file = h5py.File(dataset_folder, "r", swmr=True)
    while True:
        obj = random.choice(render_objs)
        times = random.randint(min_obj, max_obj)
        if use_hdf5:
            sample = h5_file[obj][f"{obj}_{times:04d}"]
            Im = (sample["im"][...] / 255.0).astype("float32")
            if config.use_median:
                B = sample["bgr_med"][...]
            else:
                B = sample["bgr"][...]
            B = (B / 255.0).astype("float32")
            GT = sample["GT"]
            hs_frames = []
            for ki in range(config.fmo_steps):
                gti = GT[f"image-{ki+1:06d}"][...]
                gt_batch = transforms.ToTensor()(
                    (gti / 65536.0).astype("float32")
                )
                hs_frames.append(gt_batch)
            gt_paths = torch.stack(hs_frames, 0).contiguous()
        else:
            filename = Path(dataset_folder, obj, f"{obj}_{times:04d}.png")
            gt_folder = Path(dataset_folder, obj, "GT", f"{obj}_{times:04d}")

            if config.use_median:
                bgr_path = gt_folder / "bgr_med.png"
            if not config.use_median or not bgr_path.exists():
                bgr_path = gt_folder / "bgr.png"

            if not filename.exists() or not bgr_path.exists():
                print(f"Something does not exist: {filename} or {bgr_path}")
                continue
            Im = Image.open(filename)
            B = Image.open(bgr_path)
            gt_paths = []
            for ki in range(config.fmo_steps):
                gt_paths.append(gt_folder / f"image-{ki+1:06d}.png")

        preprocess = get_transform(config.normalize)
        if use_latent_learning:
            if use_hdf5:
                print("Not implemented!")
            else:
                diffbgr_folder = Path(dataset_folder, obj, "diffbgr")
                I2 = Image.open(diffbgr_folder / f"{times:04d}_im.png")
                if config.use_median:
                    B2 = Image.open(diffbgr_folder / f"{times:04d}_bgrmed.png")
                else:
                    B2 = Image.open(diffbgr_folder / f"{times:04d}_bgr.png")
            input_batch = torch.cat(
                (
                    preprocess(Im),
                    preprocess(B),
                    preprocess(I2),
                    preprocess(B2),
                ),
                0,
            )
        else:
            input_batch = torch.cat((preprocess(Im), preprocess(B)), 0)

        return input_batch, gt_paths


def get_gt_sample(gt_paths: List[str], ti: int) -> torch.Tensor:
    GT = Image.open(gt_paths[ti])
    preprocess = transforms.ToTensor()
    gt_batch = preprocess(GT)
    return gt_batch


def get_dataset_statistics(dataset_folder: Path, config: Config) -> None:
    nobj = 0
    all_times = []
    all_objs_max = []
    for obj in config.render_objs:
        times = 0
        while True:
            filename = Path(dataset_folder, obj, f"{obj}_{times+1:04d}.png")
            bgr_path = Path(
                dataset_folder,
                obj,
                "GT",
                f"{obj}_{times+1:04d}",
                "bgr.png",
            )
            if not filename.exists() or not bgr_path.exists():
                break
            times += 1
        print(f"Object {obj} has {times} instances")
        all_times.append(times)
        if times > 0:
            nobj += 1
        if times == config.number_per_category:
            all_objs_max.append(obj)
    print(f"Number of objects {len(config.render_objs)}")
    print(f"Number of non-zero objects {nobj}")
    print(all_times)
    print(all_objs_max)
    print(len(all_objs_max))
