#!/usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from config import Config, load_config
from dataloaders.loader import get_transform
from models.encoder import EncoderCNN
from models.rendering import RenderingCNN
from train import Trainer
from utils import (
    crop_resize,
    extend_bbox,
    fmo_detect_maxarea,
    imread,
    imwrite,
    rev_crop_resize,
    rgba2hs,
)


class Runner:
    def __init__(self, config: Config, load_folder: Path):
        self.config = config

        torch.backends.cudnn.benchmark = True
        self.encoder = EncoderCNN()
        self.rendering = RenderingCNN(config)

        encoder_name = f"{Trainer.ENC_PREFIX}{Trainer.BEST_SUFFIX}.pt"
        rendering_name = f"{Trainer.RENDER_PREFIX}{Trainer.BEST_SUFFIX}.pt"
        self.encoder.load_state_dict(
            torch.load(load_folder / encoder_name, map_location="cpu")
        )
        self.rendering.load_state_dict(
            torch.load(load_folder / rendering_name, map_location="cpu")
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = self.encoder.to(self.device)
        self.rendering = self.rendering.to(self.device)

        self.encoder.eval()
        self.rendering.eval()

    def process_image(
        self,
        im_path: Path,
        bgr_path: Path,
        output_folder: Path,
        steps: int,
    ) -> None:
        im = imread(str(im_path))
        bgr = imread(str(bgr_path))
        tsr = self._run_defmo(im, bgr, steps)
        # generate results
        out = cv2.VideoWriter(
            str(output_folder / "tsr.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            6,
            (im.shape[1], im.shape[0]),
            True,
        )
        for ki in range(steps):
            imwrite(tsr[..., ki], str(output_folder / f"tsr{ki}.png"))
            out.write((tsr[:, :, [2, 1, 0], ki] * 255).astype(np.uint8))
        out.release()

    def process_video(
        self,
        video_path: Path,
        output_folder: Path,
        median: int,
        steps: int,
    ) -> None:
        # estimate initial background
        Ims = []
        cap = cv2.VideoCapture(str(video_path))
        while cap.isOpened():
            ret, frame = cap.read()
            Ims.append(frame)
            if len(Ims) >= median:
                break
        bgr = np.median(np.asarray(Ims) / 255, 0)[:, :, [2, 1, 0]]

        # run DeFMO
        out = cv2.VideoWriter(
            str(output_folder / "tsr.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            6,
            (bgr.shape[1], bgr.shape[0]),
            True,
        )
        tsr0: Optional[np.ndarray] = None
        frmi = 0
        while cap.isOpened():
            if frmi < median:
                frame = Ims[frmi]
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                Ims = Ims[1:]
                Ims.append(frame)
                # update background (running median)
                bgr = np.median(np.asarray(Ims) / 255, 0)[:, :, [2, 1, 0]]
            frmi += 1
            im = frame[:, :, [2, 1, 0]] / 255
            tsr = self._run_defmo(im, bgr, steps)
            if frmi == 1:
                tsr0 = tsr
                continue
            if frmi == 2:
                assert tsr0 is not None
                forward = np.min(
                    [
                        np.mean((tsr0[..., -1] - tsr[..., -1]) ** 2),
                        np.mean((tsr0[..., -1] - tsr[..., 0]) ** 2),
                    ]
                )
                backward = np.min(
                    [
                        np.mean((tsr0[..., 0] - tsr[..., -1]) ** 2),
                        np.mean((tsr0[..., 0] - tsr[..., 0]) ** 2),
                    ]
                )
                if backward < forward:
                    # reverse time direction for better alignment
                    tsr0 = tsr0[..., ::-1]
                    assert tsr0 is not None
                for ki in range(steps):
                    out.write(
                        (tsr0[:, :, [2, 1, 0], ki] * 255).astype(np.uint8)
                    )

            assert tsr0 is not None
            if np.mean((tsr0[..., -1] - tsr[..., -1]) ** 2) < np.mean(
                (tsr0[..., -1] - tsr[..., 0]) ** 2
            ):
                # reverse time direction for better alignment
                tsr = tsr[..., ::-1]

            for ki in range(steps):
                out.write((tsr[:, :, [2, 1, 0], ki] * 255).astype(np.uint8))
            tsr0 = tsr
        cap.release()
        out.release()

    def _run_defmo(
        self,
        im: np.ndarray,
        bgr: np.ndarray,
        steps: int,
    ) -> np.ndarray:
        preprocess = get_transform()
        bbox, radius = fmo_detect_maxarea(im, bgr, maxarea=0.03)
        bbox = extend_bbox(
            bbox.copy(),
            4 * np.max(radius),
            self.config.resolution_y / self.config.resolution_x,
            im.shape,
        )
        im_crop = crop_resize(
            im, bbox, (self.config.resolution_x, self.config.resolution_y)
        )
        bgr_crop = crop_resize(
            bgr, bbox, (self.config.resolution_x, self.config.resolution_y)
        )
        input_batch = (
            torch.cat((preprocess(im_crop), preprocess(bgr_crop)), 0)
            .to(self.device)
            .unsqueeze(0)
            .float()
        )
        with torch.no_grad():
            latent = self.encoder(input_batch)
            times = torch.linspace(0, 1, steps).to(self.device)
            renders = self.rendering(latent, times[None])
        renders_rgba = (
            renders[0].data.cpu().detach().numpy().transpose(2, 3, 1, 0)
        )
        tsr_crop = rgba2hs(renders_rgba, bgr_crop)
        tsr = rev_crop_resize(tsr_crop, bbox, bgr.copy())
        tsr[tsr > 1] = 1
        tsr[tsr < 0] = 0
        return tsr


def main(args: Namespace) -> None:
    config = load_config(args.config)

    output_folder = args.output_folder.expanduser()
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    runner = Runner(config, args.load_folder.expanduser())
    if args.im is not None and args.bgr is not None:
        runner.process_image(
            args.im,
            args.bgr,
            output_folder,
            args.steps,
        )
    elif args.video is not None:
        runner.process_video(
            args.video,
            output_folder,
            args.median,
            args.steps,
        )
    else:
        print("You should either provide both --im and --bgr, or --video.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run the DeFMO model on input images/videos",
        epilog="You must either give both --im and --bgr, or give --video",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "load_folder", type=Path, help="Path from where to load saved models"
    )
    parser.add_argument("--im", type=Path, help="Path to the input image")
    parser.add_argument(
        "--bgr", type=Path, help="Path to the input background"
    )
    parser.add_argument("--video", type=Path, help="Path to the input video")
    parser.add_argument(
        "--config", type=Path, help="Path to the TOML hyper-param config"
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default="output",
        help="Path where to dump outputs",
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--median", type=int, default=7)
    main(parser.parse_args())
