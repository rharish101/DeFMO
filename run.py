#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from config import Config, load_config
from dataloaders.loader import get_transform
from models.encoder import EncoderCNN
from models.rendering import RenderingCNN
from utils import (
    crop_resize,
    extend_bbox,
    fmo_detect_maxarea,
    imread,
    imwrite,
    rev_crop_resize,
    rgba2hs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--im", default=None, required=False)
    parser.add_argument("--bgr", default=None, required=False)
    parser.add_argument("--video", default=None, required=False)
    parser.add_argument("--steps", default=24, required=False)
    parser.add_argument("--models", default="saved_models", required=False)
    parser.add_argument("--output", default="output", required=False)
    parser.add_argument("--median", default=7, required=False)
    parser.add_argument(
        "--config", type=Path, help="Path to the TOML hyper-param config"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device(
        "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
    )
    torch.backends.cudnn.benchmark = True
    encoder = EncoderCNN()
    rendering = RenderingCNN(config)

    if torch.cuda.is_available():
        encoder.load_state_dict(
            torch.load(os.path.join(args.models, "encoder_best.pt"))
        )
        rendering.load_state_dict(
            torch.load(os.path.join(args.models, "rendering_best.pt"))
        )
    else:
        encoder.load_state_dict(
            torch.load(
                os.path.join(args.models, "encoder_best.pt"),
                map_location=torch.device("cpu"),
            )
        )
        rendering.load_state_dict(
            torch.load(
                os.path.join(args.models, "rendering_best.pt"),
                map_location=torch.device("cpu"),
            )
        )

    encoder = encoder.to(device)
    rendering = rendering.to(device)

    encoder.train(False)
    rendering.train(False)

    if args.im is not None and args.bgr is not None:
        im = imread(args.im)
        bgr = imread(args.bgr)
        tsr = run_defmo(
            config, im, bgr, rendering, encoder, args.steps, device
        )
        # generate results
        out = cv2.VideoWriter(
            os.path.join(args.output, "tsr.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            6,
            (im.shape[1], im.shape[0]),
            True,
        )
        for ki in range(args.steps):
            imwrite(
                tsr[..., ki], os.path.join(args.output, "tsr{}.png".format(ki))
            )
            out.write((tsr[:, :, [2, 1, 0], ki] * 255).astype(np.uint8))
        out.release()
    elif args.video is not None:
        # estimate initial background
        Ims = []
        cap = cv2.VideoCapture(args.video)
        while cap.isOpened():
            ret, frame = cap.read()
            Ims.append(frame)
            if len(Ims) >= args.median:
                break
        bgr = np.median(np.asarray(Ims) / 255, 0)[:, :, [2, 1, 0]]

        # run DeFMO
        out = cv2.VideoWriter(
            os.path.join(args.output, "tsr.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            6,
            (bgr.shape[1], bgr.shape[0]),
            True,
        )
        tsr0: Optional[np.ndarray] = None
        frmi = 0
        while cap.isOpened():
            if frmi < args.median:
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
            tsr = run_defmo(
                config, im, bgr, rendering, encoder, args.steps, device
            )
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
                for ki in range(args.steps):
                    out.write(
                        (tsr0[:, :, [2, 1, 0], ki] * 255).astype(np.uint8)
                    )

            assert tsr0 is not None
            if np.mean((tsr0[..., -1] - tsr[..., -1]) ** 2) < np.mean(
                (tsr0[..., -1] - tsr[..., 0]) ** 2
            ):
                # reverse time direction for better alignment
                tsr = tsr[..., ::-1]

            for ki in range(args.steps):
                out.write((tsr[:, :, [2, 1, 0], ki] * 255).astype(np.uint8))
            tsr0 = tsr
        cap.release()
        out.release()
    else:
        print("You should either provide both --im and --bgr, or --video.")


def run_defmo(
    config: Config,
    im: np.ndarray,
    bgr: np.ndarray,
    rendering: torch.nn.Module,
    encoder: torch.nn.Module,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    preprocess = get_transform()
    bbox, radius = fmo_detect_maxarea(im, bgr, maxarea=0.03)
    bbox = extend_bbox(
        bbox.copy(),
        4 * np.max(radius),
        config.resolution_y / config.resolution_x,
        im.shape,
    )
    im_crop = crop_resize(im, bbox, (config.resolution_x, config.resolution_y))
    bgr_crop = crop_resize(
        bgr, bbox, (config.resolution_x, config.resolution_y)
    )
    input_batch = (
        torch.cat((preprocess(im_crop), preprocess(bgr_crop)), 0)
        .to(device)
        .unsqueeze(0)
        .float()
    )
    with torch.no_grad():
        latent = encoder(input_batch)
        times = torch.linspace(0, 1, steps).to(device)
        renders = rendering(latent, times[None])
    renders_rgba = renders[0].data.cpu().detach().numpy().transpose(2, 3, 1, 0)
    tsr_crop = rgba2hs(renders_rgba, bgr_crop)
    tsr = rev_crop_resize(tsr_crop, bbox, bgr.copy())
    tsr[tsr > 1] = 1
    tsr[tsr < 0] = 0
    return tsr


if __name__ == "__main__":
    main()
