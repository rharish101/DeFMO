import os
import random

import h5py
import torch
from PIL import Image
from torchvision import transforms


class ShapeBlurDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_folder,
        config,
        render_objs,
        number_per_category,
        do_augment=False,
        use_latent_learning=False,
    ):
        self.timestamps = torch.linspace(0, 1, config.fmo_steps)
        self.dataset_folder = dataset_folder
        self.config = config
        self.render_objs = render_objs
        self.number_per_category = number_per_category
        self.do_augment = do_augment
        self.use_latent_learning = use_latent_learning

    def __len__(self):
        return len(self.render_objs) * self.number_per_category

    def __getitem__(self, index):
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
            inputs = transforms.functional.resize(
                inputs, [self.config.train_res_x, self.config.train_res_y]
            )
            hs_frames = transforms.functional.resize(
                hs_frames, [self.config.train_res_x, self.config.train_res_y]
            )

        return inputs, times, hs_frames, times_left


def get_transform(normalize):
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
    dataset_folder,
    config,
    render_objs,
    max_obj,
    min_obj=1,
    use_latent_learning=False,
):
    use_hdf5 = "hdf5" in dataset_folder
    if use_hdf5:
        h5_file = h5py.File(dataset_folder, "r", swmr=True)
    while True:
        obj = random.choice(render_objs)
        times = random.randint(min_obj, max_obj)
        if use_hdf5:
            sample = h5_file[obj]["{}_{:04d}".format(obj, times)]
            Im = (sample["im"][...] / 255.0).astype("float32")
            if config.use_median:
                B = sample["bgr_med"][...]
            else:
                B = sample["bgr"][...]
            B = (B / 255.0).astype("float32")
            GT = sample["GT"]
            hs_frames = []
            for ki in range(config.fmo_steps):
                gti = GT["image-{:06d}".format(ki + 1)][...]
                gt_batch = transforms.ToTensor()(
                    (gti / 65536.0).astype("float32")
                )
                hs_frames.append(gt_batch)
            gt_paths = torch.stack(hs_frames, 0).contiguous()
        else:
            filename = os.path.join(
                dataset_folder, obj, "{}_{:04d}.png".format(obj, times)
            )
            if config.use_median:
                bgr_path = os.path.join(
                    dataset_folder,
                    obj,
                    "GT",
                    "{}_{:04d}".format(obj, times),
                    "bgr_med.png",
                )
            if not config.use_median or not os.path.exists(bgr_path):
                bgr_path = os.path.join(
                    dataset_folder,
                    obj,
                    "GT",
                    "{}_{:04d}".format(obj, times),
                    "bgr.png",
                )

            if not os.path.exists(filename) or not os.path.exists(bgr_path):
                print(
                    "Something does not exist: {} or {}".format(
                        filename, bgr_path
                    )
                )
                continue
            Im = Image.open(filename)
            B = Image.open(bgr_path)
            gt_paths = []
            for ki in range(config.fmo_steps):
                gt_paths.append(
                    os.path.join(
                        dataset_folder,
                        obj,
                        "GT",
                        "{}_{:04d}".format(obj, times),
                        "image-{:06d}.png".format(ki + 1),
                    )
                )

        preprocess = get_transform(config.normalize)
        if use_latent_learning:
            if use_hdf5:
                print("Not implemented!")
            else:
                I2 = Image.open(
                    os.path.join(
                        dataset_folder,
                        obj,
                        "diffbgr",
                        "{:04d}_im.png".format(times),
                    )
                )
                if config.use_median:
                    B2 = Image.open(
                        os.path.join(
                            dataset_folder,
                            obj,
                            "diffbgr",
                            "{:04d}_bgrmed.png".format(times),
                        )
                    )
                else:
                    B2 = Image.open(
                        os.path.join(
                            dataset_folder,
                            obj,
                            "diffbgr",
                            "{:04d}_bgr.png".format(times),
                        )
                    )
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


def get_gt_sample(gt_paths, ti):
    GT = Image.open(gt_paths[ti])
    preprocess = transforms.ToTensor()
    gt_batch = preprocess(GT)
    return gt_batch


def get_dataset_statistics(dataset_folder, config):
    nobj = 0
    all_times = []
    all_objs_max = []
    for obj in config.render_objs:
        times = 0
        while True:
            filename = os.path.join(
                dataset_folder, obj, "{}_{:04d}.png".format(obj, times + 1)
            )
            bgr_path = os.path.join(
                dataset_folder,
                obj,
                "GT",
                "{}_{:04d}".format(obj, times + 1),
                "bgr.png",
            )
            if not os.path.exists(filename) or not os.path.exists(bgr_path):
                break
            times += 1
        print("Object {} has {} instances".format(obj, times))
        all_times.append(times)
        if times > 0:
            nobj += 1
        if times == config.number_per_category:
            all_objs_max.append(obj)
    print("Number of objects {}".format(len(config.render_objs)))
    print("Number of non-zero objects {}".format(nobj))
    print(all_times)
    print(all_objs_max)
    print(len(all_objs_max))
