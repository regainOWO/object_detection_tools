"""
脚本的作用如下:
灰度图转RGB图像，再维度上从1变成3，扩充的方式是进行复制
"""
import argparse
import os
import os.path as osp

from PIL.Image import init, EXTENSION
import cv2
from tqdm import tqdm

if not len(EXTENSION):
    init()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default=r'D:\dataset\plane_and_vehicle\val\images', help='input dataset dir')
    parser.add_argument('--output-dir', type=str, default=None, help='outptu directory, default is overwrite')
    opt = parser.parse_args()
    return opt


def run(image_dir, output_dir):
    assert osp.isdir(image_dir), f'文件夹不存在: {image_dir}'
    if output_dir is None:
        output_dir = image_dir
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    for basename in os.listdir(image_dir):
        filename, suffix = osp.splitext(basename)
        if suffix not in EXTENSION:
            continue
        image_file = osp.join(image_dir, basename)
        image_files.append(image_file)

    for image_file in tqdm(image_files, total=len(image_files)):
        image_basename = osp.basename(image_file)
        dst_file = osp.join(output_dir, image_basename)
        image = cv2.imread(image_file)
        cv2.imwrite(dst_file, image)


if __name__ == "__main__":
    opt = parse_opt()
    run(image_dir=opt.image_dir,
        output_dir=opt.output_dir)
