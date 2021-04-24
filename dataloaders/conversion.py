#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Optional

import cv2
import h5py
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the images, convert them to the network input and "
        "store it as a dataset"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing the image pairs",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        help="Output dataset file",
    )
    parser.add_argument(
        "-c",
        dest="compress",
        action="store_true",
        help="Compress the data with lzf compression",
    )

    args = parser.parse_args()

    if args.output_file is None:
        output_file = args.input_dir.parent / args.input_dir.name + ".hdf5"
    else:
        output_file = args.output_file

    if args.compress:
        compression_type: Optional[str] = "lzf"
    else:
        compression_type = None

    all_filenames = [f for f in args.input_dir.glob("*")]

    # create h5 file where all the data will be stored
    if output_file.exists():
        answer = input(f"File '{output_file}' exists. Delete? (Y/n): ")
        if answer.strip().lower() == "n":
            return
        output_file.unlink()

    print(Path.cwd())
    output_file = h5py.File(output_file)

    for obj_type in tqdm(all_filenames):
        # load best aligned images

        output_file.create_group(obj_type.name)
        all_objects = [f for f in obj_type.glob("*") if ".png" in f.name]

        for obj_name in tqdm(all_objects):
            name = "_".join(obj_name.stem.split("."))
            output_file[obj_type.name].create_group(name)

            im_path = obj_name
            bgr_folder = Path(obj_type, "GT", name)
            bgr_path = bgr_folder / "bgr.png"
            bgrmed_path = bgr_folder / "bgr_med.png"

            # load the images
            im = cv2.imread(str(im_path), -1)
            bgr = cv2.imread(str(bgr_path), -1)
            bgr_med = cv2.imread(str(bgrmed_path), -1)

            # adding data to dataset file
            output_file[obj_type.name][name].create_dataset(
                "im", data=im, compression=compression_type
            )
            output_file[obj_type.name][name].create_dataset(
                "bgr", data=bgr, compression=compression_type
            )
            output_file[obj_type.name][name].create_dataset(
                "bgr_med", data=bgr_med, compression=compression_type
            )

            all_gt = [f for f in bgr_folder.glob("*") if "image" in f.name]
            output_file[obj_type.name][name].create_group("GT")
            for gtname in tqdm(all_gt):
                gtsavename = "_".join(gtname.stem.split("."))
                gtim = cv2.imread(str(gtname), -1)
                output_file[obj_type.name][name]["GT"].create_dataset(
                    gtsavename, data=gtim, compression=compression_type
                )

    output_file.close()


if __name__ == "__main__":
    main()
