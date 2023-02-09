import argparse
import glob
import logging
import os
import os.path as osp
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--maxsize', type=int, default=1024,
                        help='scale image result max pixel size, default is 1024')
    parser.add_argument('--image-dirname', type=str, default='images',
                        help='the image directory name under the dataset directory, default is images')
    parser.add_argument('--label-dirname', type=str, default='labels',
                        help='the dota label directory name under the dataset directory, default is labelTxt')
    parser.add_argument('--output-path', type=str, default=None,
                        help='output path, default is split directory under the your workingdirectory')
    parser.add_argument('--save-ext', type=str, default='.png',
                        help='split image save suffix, default is .png')
    opt = parser.parse_args()
    return opt


def get_path_basename(path):
    p = Path(path)
    return p.name.split(p.suffix)[0]


def proprotional_scale_dota(image_file, label_file, o_image_file, o_label_file, maxsize, ext):
    # dota
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    polys = np.array(lines[:, :8], dtype=np.float32)
    classnames = lines[:, 8]
    difficults = lines[:, 9]
    img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), -1)
    h0, w0 = img.shape[:2]
    rate = min(maxsize / h0, maxsize / w0)

    rate = min(rate, 1.0)     # not scale up
    if rate != 1:
        r_img = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        r_polys = polys * rate
        np.random
    else:
        r_img = img
        r_polys = polys
    cv2.imencode(ext, r_img)[1].tofile(o_image_file)
    content_lines = []
    for r_poly, classname, difficult in zip(r_polys, classnames, difficults):
        content_line = " ".join(str(a) for a in r_poly) + " " + str(classname) + " " + str(difficult) + '\n'
        content_lines.append(content_line)
    with open(o_label_file, 'w', encoding='utf-8') as f:
        f.writelines(content_lines)


def proprotional_scale(image_file, label_file, o_image_file, o_label_file, maxsize, ext):
    # normalized hbb yolo
    shutil.copy(label_file, o_label_file)
    img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), -1)
    h0, w0 = img.shape[:2]
    rate = min(maxsize / h0, maxsize / w0)

    rate = min(rate, 1.0)     # not scale up
    if rate != 1:
        r_img = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
    else:
        r_img = img
    cv2.imencode(ext, r_img)[1].tofile(o_image_file)


def run(dataset_dir, maxsize, image_dirname, label_dirname, output_path, ext):
    image_dir = osp.join(dataset_dir, image_dirname)
    label_dir = osp.join(dataset_dir, label_dirname)
    label_files = [osp.join(label_dir, filename) for filename in os.listdir(label_dir)]
    output_path = output_path if output_path else osp.join(dataset_dir, f"scale_{str(maxsize)}")
    output_image_dir = osp.join(output_path, image_dirname)
    output_label_dir = osp.join(output_path, label_dirname)
    if osp.exists(output_path):
        logging.info(f"remove outdate directory {output_path}...")
        shutil.rmtree(output_path)
    os.makedirs(output_image_dir)
    os.makedirs(output_label_dir)

    for label_file in tqdm(label_files, total=len(label_files)):
        # 在图片文件夹中寻找相同文件名称的图片文件
        file_basename = get_path_basename(label_file)
        try:
            image_file = glob.glob(osp.join(image_dir, file_basename + '.*'))[0]
        except Exception as e:
            assert 0, f"please make sure the folder: {image_dir} have correct image named like {file_basename}"
        assert image_file.rsplit('.')[-1] in IMG_FORMATS, f'find file: {image_file}, but is not an image format'
        o_image_file = osp.join(output_image_dir, Path(image_file).with_suffix(ext).name)
        o_label_file = osp.join(output_label_dir, Path(label_file).name)
        proprotional_scale(image_file, label_file, o_image_file, o_label_file, maxsize, ext)

    # print msg
    print(f"scale dataset: {Path(dataset_dir).name} Success!!! maxsize is {maxsize}")
    print(f"scale result is in folder {output_path}")


if __name__ == '__main__':
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        maxsize=opt.maxsize,
        image_dirname=opt.image_dirname,
        label_dirname=opt.label_dirname,
        output_path=opt.output_path,
        ext=opt.save_ext)