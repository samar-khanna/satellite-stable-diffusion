import json
import argparse
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import webdataset as wds


if __name__ == "__main__":
    parser = argparse.ArgumentParser('create web dataset')
    parser.add_argument('--csv_path', type=str, help='Path to data csv path')
    parser.add_argument('--metadata_path', type=str, help='Path to dataset metadata')
    parser.add_argument('--dest_path', type=str, help='Path to to destination .tar file.')
    args = parser.parse_args()

    data_info = pd.read_csv(args.csv_path)
    image_arr = np.asarray(data_info.iloc[:, 1])
    label_arr = np.asarray(data_info.iloc[:, 0])

    sink = wds.TarWriter(args.dest_path)

    pbar = tqdm(zip(image_arr, label_arr), total=len(image_arr))
    for i, (im_path, label) in enumerate(pbar):
        img = Image.open(im_path)

        meta_path = im_path.replace('_crop_0.jpg', '.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        assert len(img.shape) == 3, f"Image {i} has incorrect shape."
        assert type(label) == int

        components = im_path.split(os.path.sep)
        cls = components[-3]  # eg: airport
        instance = components[-2]  # eg: airport_0
        imgid = components[-1].replace("_crop_0", "")  # eg: airport_0_0_rgb.jpg

        sink.write({
            "__key__": f"{cls}-{instance}-{imgid}",
            "input.jpg": img,
            "output.cls": label,
            "metadata.json": metadata,
        })

