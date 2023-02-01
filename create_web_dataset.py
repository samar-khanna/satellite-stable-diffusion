import json
import argparse
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import webdataset as wds

from concurrent.futures import ThreadPoolExecutor, as_completed


def create_output_dict(im_path, label):
    with Image.open(im_path) as im:
        img_arr = np.array(im)

    meta_path = im_path.replace('_crop_0.jpg', '.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # assert len(img.shape) == 3, f"Image {i} has incorrect shape."
    # assert type(label) == int

    components = im_path.split(os.path.sep)
    cls = components[-3]  # eg: airport
    instance = components[-2]  # eg: airport_0
    imgid = components[-1].replace("_crop_0.jpg", "")  # eg: airport_0_0_rgb.jpg

    out = {
            "__key__": f"{cls}-{instance}-{imgid}",
            "input.png": img_arr,
            "output.cls": label,
            "metadata.json": metadata,
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser('create web dataset')
    parser.add_argument('--csv_path', type=str, help='Path to data csv path')
    parser.add_argument('--metadata_path', type=str, help='Path to dataset metadata')
    parser.add_argument('--dest_path', type=str, help='Path to to destination .tar file.')
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    data_info = pd.read_csv(args.csv_path)
    image_arr = np.asarray(data_info.iloc[:, 1])
    label_arr = np.asarray(data_info.iloc[:, 0])

    sink = wds.TarWriter(args.dest_path)

    with ThreadPoolExecutor(max_workers=args.num_workers) as exec:
        future_to_out = {exec.submit(create_output_dict, im_path, label): (im_path, label)
                         for im_path, label in zip(image_arr, label_arr)}

        for future in tqdm(as_completed(future_to_out), total=len(image_arr)):
            try:
                out_dict = future.result()
            except Exception as e:
                raise e

            sink.write(out_dict)
            out_dict['input.png'].close()

    # pbar = tqdm(zip(image_arr, label_arr), total=len(image_arr))
    # for i, (im_path, label) in enumerate(pbar):
    #     img = Image.open(im_path)
    #
    #     meta_path = im_path.replace('_crop_0.jpg', '.json')
    #     with open(meta_path, 'r') as f:
    #         metadata = json.load(f)
    #
    #     # assert len(img.shape) == 3, f"Image {i} has incorrect shape."
    #     # assert type(label) == int
    #
    #     components = im_path.split(os.path.sep)
    #     cls = components[-3]  # eg: airport
    #     instance = components[-2]  # eg: airport_0
    #     imgid = components[-1].replace("_crop_0.jpg", "")  # eg: airport_0_0_rgb.jpg
    #
    #     sink.write({
    #         "__key__": f"{cls}-{instance}-{imgid}",
    #         "input.png": img,
    #         "output.cls": label,
    #         "metadata.json": metadata,
    #     })

