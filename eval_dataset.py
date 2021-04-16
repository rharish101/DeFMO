import numpy as np
from argparse import ArgumentParser, Namespace
import torch
from pathlib import Path

from models.encoder import EncoderCNN
from models.rendering import RenderingCNN
from dataloaders.tbd_loader import evaluate_on
from config import load_config
from helpers.paper_helpers import get_figure_images

def main(args: Namespace) -> None:
    config = load_config(args.config)
    print(torch.__version__)

    gpu_id = 0
    train_mode = False

    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    print(device)
    torch.backends.cudnn.benchmark = True
    encoder = EncoderCNN()
    rendering = RenderingCNN(config)

    if torch.cuda.is_available():
        encoder.load_state_dict(torch.load(args.run_folder / 'encoder_best.pt'))
        rendering.load_state_dict(torch.load(args.run_folder / 'rendering_best.pt'))
    else:
        encoder.load_state_dict(torch.load(args.run_folder / 'encoder_best.pt',map_location=torch.device('cpu')))
        rendering.load_state_dict(torch.load(args.run_folder / 'rendering_best.pt',map_location=torch.device('cpu')))
        
    encoder = encoder.to(device)
    rendering = rendering.to(device)

    encoder.train(train_mode)
    rendering.train(train_mode)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    rendering_params = sum(p.numel() for p in rendering.parameters())
    print('Encoder params {:2f}M, rendering params {:2f}M'.format(encoder_params/1e6,rendering_params/1e6))

    ## full evaluation on datasets
    if False:
        datasets = ['tbd','tbd3d','tbdfalling','wildfmo','youtube']
        evaluate_on(encoder, rendering, device, datasets[2])

    ## run on falling objects 
    if True:
        get_figure_images(args.dataset_folder, encoder, rendering, device, 'tbdfalling', 2, 31+2, results_mode=True, n_occ=7)

    ## run on tbd-3d
    if False:
        get_figure_images(args.dataset_folder, encoder, rendering, device, 'tbd3d', 1, 13, results_mode=True)
        get_figure_images(args.dataset_folder, encoder, rendering, device, 'tbd3d', 2, 1, results_mode=True)
        get_figure_images(args.dataset_folder, encoder, rendering, device, 'tbd3d', 6, 23, results_mode=True)
        get_figure_images(args.dataset_folder, encoder, rendering, device, 'tbd3d', 7, 19, results_mode=True)

    ## run on tbd
    if False:
        get_figure_images(args.dataset_folder, encoder, rendering, device, 'tbd', 0, 1, results_mode=True, n_occ=7)
        get_figure_images(args.dataset_folder, encoder, rendering, device, 'tbd', 'volleyball', 43, results_mode=True, n_occ=7)

    # run on synthetic data
    if False:
        objid = np.nonzero([temp == 'mug' for temp in config.render_objs_train])[0][0]
        get_figure_images(args.dataset_folder, encoder, rendering, device, 'train', objid, 123, results_mode=True, n_occ=7)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "dataset_folder", type=Path, help="Path to the dataset"
    )
    parser.add_argument(
        "run_folder",
        type=Path,
        help="Path to the run whose saved models are to be loaded",
    )
    parser.add_argument(
        "--config", type=Path, help="Path to the TOML hyper-param config"
    )
    main(parser.parse_args())
