# Evaluation, Training, Demo, and Inference of DeFMO 

### DeFMO: Deblurring and Shape Recovery of Fast Moving Objects (arxiv 2020)
#### Denys Rozumnyi, Martin R. Oswald, Vittorio Ferrari, Jiri Matas, Marc Pollefeys

### Qualitative results: https://www.youtube.com/watch?v=pmAynZvaaQ4


<img src="example/results_defmo.png" width="500">

### Pre-trained models

The pre-trained DeFMO model as reported in the paper is available here: https://polybox.ethz.ch/index.php/s/M06QR8jHog9GAcF. Put them into ./saved_models sub-folder.

### Inference
For generating video temporal super-resolution:
```bash
python run.py --video example/falling_pen.avi
```

For generating temporal super-resolution of a single frame with the given background:
```bash
python run.py --im example/im.png --bgr example/bgr.png
```

### Training
Set up all paths in main_settings.py and run
```bash
python train.py
```

Reference
------------
If you use this repository, please cite the following publication ( https://arxiv.org/abs/2012.00595 ):

```bibtex
@inproceedings{defmo,
  author = {Denys Rozumnyi and Martin R. Oswald and Vittorio Ferrari and Jiri Matas and Marc Pollefeys},
  title = {DeFMO: Deblurring and Shape Recovery of Fast Moving Objects},
  booktitle = {arxiv},
  address = {online},
  month = dec,
  year = {2020}
}
```
