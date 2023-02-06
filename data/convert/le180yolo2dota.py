"""
此脚本的主要内容:
将le180旋转框的数据格式转换成dota格式，支持水平框和旋转框
cls_id cx cy longsize shortside angle ========> x1 y1 x2 y2 x3 y3 x4 y4 classname diffcult

diffcult：默认为0
angle: 范围为[0, 180)

使用方法，需要输入三个参数：
--dataset-dir: 数据集的根路径
--image-dirname: 图片的文件夹名称，默认值为images（图片文件夹需要放在数据集的根文件夹中）
--label-dirname: yolo obb标签的文件夹名称，默认值为labels_obb（标签文件夹需要放在数据集的根文件夹中）

输入的路径：文件夹名称固定值为labelTxt (放在数据集的根文件夹中）
"""

import argparse
import glob
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--image-dirname', type=str, default='images', help='the image directory name under the dataset directory')
    parser.add_argument('--label-dirname', type=str, default='labels_obb', help='the yolo obb label directory name under the dataset directory')
    opt = parser.parse_args()
    return opt


def le180yolo2dota(in_file, out_file, classes, img_wh):
    diffcult = 0
    with open(in_file, mode='r', encoding='utf-8') as f:
        labels = [str(x).strip().split(' ') for x in f.readlines()]
    labels = np.array(labels, dtype=np.float64).reshape(-1, 6)
    cls_ids = labels[:, [0]]
    rboxs = labels[:, 1:].copy()
    # 去归一化，变回原图上的尺度
    rboxs[:, [0, 2]] *= img_wh[0]
    rboxs[:, [1, 3]] *= img_wh[1]
    # rboxs[:, -1] = (rboxs[:, -1] - 90) * np.pi / 180   # angle∈[-pi/2, pi/2)，转换一下空间
    # polys = rbox2poly(rboxs)
    polys = [np.float32(cv2.boxPoints(((x[0], x[1]), (x[2], x[3]), x[4]))).reshape(8,) for x in rboxs]
    # polys = [longsideformat2poly(*x) for x in rboxs]

    content_lines = []
    for cls_id, poly in zip(cls_ids, polys):
        line_one = ' '.join(["%.1f" % a for a in poly]) + ' ' + classes[int(cls_id)] + ' ' + str(diffcult) + '\n'
        content_lines.append(line_one)
    # 将内容写入文件
    with open(out_file, mode='w', encoding='utf-8') as f:
        f.writelines(content_lines)


def run(dataset_dir, image_dirname='images', label_dirname='labels_obb'):
    class_filepath = osp.join(dataset_dir, 'classes.txt')
    assert os.path.exists(class_filepath), f'please make sure the classes.txt is in path: {dataset_dir}'
    with open(class_filepath) as f:
        classes = [str(x).strip() for x in f.readlines()]
    label_dir = osp.join(dataset_dir, label_dirname)
    image_dir = osp.join(dataset_dir, image_dirname)
    filenames = os.listdir(label_dir)
    label_files = [osp.join(label_dir, x) for x in filenames]
    save_dir = osp.join(dataset_dir, 'labelTxt')
    os.makedirs(save_dir, exist_ok=True)

    for i, label_file in tqdm(enumerate(label_files), total=len(label_files)):
        out_file = osp.join(save_dir, Path(label_file).name)
        # 在图片文件夹中寻找相同文件名称的图片文件
        image_file = glob.glob(osp.join(image_dir, filenames[i].rsplit('.')[0]) + '.*')[0]
        assert image_file.rsplit('.')[-1] in IMG_FORMATS, f'find file: {image_file}, but is not an image format'
        img_wh = Image.open(image_file).size
        le180yolo2dota(label_file, out_file, classes, img_wh)


if __name__ == '__main__':
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        image_dirname=opt.image_dirname,
        label_dirname=opt.label_dirname)
    print(f"convert dataset: {Path(opt.dataset_dir).name} le180yolo obb label to dota Success!!!")
    print(f"convert label is in folder labelTxt")