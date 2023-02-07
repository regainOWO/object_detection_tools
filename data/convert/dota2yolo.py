"""
此脚本的主要内容:
将dota的数据集格式转换成yolo格式，取包裹住多边形最小的矩形框
x1 y1 x2 y2 x3 y3 x4 y4 classname diffcult ========> cls_id cx cy w h

使用方法，需要输入三个参数：
--dataset-dir: 数据集的根路径
--image-dirname: 图片的文件夹名称，默认值为images（图片文件夹需要放在数据集的根文件夹中）
--label-dirname: dota标签的文件夹名称，默认值为labelTxt（标签文件夹需要放在数据集的根文件夹中）

输入的路径：文件夹名称固定值为labels (放在数据集的根文件夹中）
"""

import argparse
import glob
import logging
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
    parser.add_argument('--label-dirname', type=str, default='labelTxt', help='the dota label directory name under the dataset directory')
    opt = parser.parse_args()
    return opt


def poly2hbb(polys):
    """Convert polygons to horizontal bboxes.

    Args:
        polys (np.array): Polygons with shape (N, 8)

    Returns:
        np.array: Horizontal bboxes. xyxy
    """
    shape = polys.shape
    polys = polys.reshape(*shape[:-1], shape[-1] // 2, 2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return np.concatenate([lt_point, rb_point], axis=-1)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def dota2yolo(in_file, out_file, classes, img_wh):
    with open(in_file, mode='r', encoding='utf-8') as f:
        labels = [str(x).strip().split(' ') for x in f.readlines()]
    polys, cls_ids = [], []
    for i, label in enumerate(labels):
        class_name = label[-2]
        difficult = int(label[-1])
        if class_name not in classes:
            print(f"ignore {in_file} line {i}, because label name not in classes.txt")
            continue
        # elif difficult > 1:
        #     print(f"ignore {in_file} line {i}, because difficult is 2")
        #     continue
        polys.append(label[:-2])
        cls_ids.append(classes.index(class_name))

    content_lines = []
    if len(polys):
        # 取包裹住object的水平框
        """
        rects = []
        for poly in polys:
            poly = np.float32(poly).reshape((4, 2))
            # 这里生成的是左上角的坐标，不是中心点的坐标
            rect = cv2.boundingRect(poly)  # (x, y, w, h)
            x = rect[0] + 0.5 * rect[2]  # x + 0.5w
            y = rect[1] + 0.5 * rect[3]  # y + 0.5h
            rects.append([x, y, rect[2], rect[3]])
        rects = np.array(rects, dtype=np.float32)
        """
        content_lines = []
        polys = np.array(polys, dtype=np.float32).reshape(-1, 8)
        rects = poly2hbb(polys)
        rects = xyxy2xywh(rects)
        # 归一化
        rects[:, [0, 2]] /= img_wh[0]
        rects[:, [1, 3]] /= img_wh[1]

        for cls_id, rect in zip(cls_ids, rects):
            line_one = str(cls_id) + ' ' + ' '.join([str(a) for a in rect]) + '\n'
            content_lines.append(line_one)

    if len(content_lines) == 0:
        logging.warning(f'no lines to save, the file: {out_file} is empty')

    # 将内容写入文件
    with open(out_file, mode='w', encoding='utf-8') as f:
        f.writelines(content_lines)


def run(dataset_dir, image_dirname='images', label_dirname='labelTxt'):
    class_filepath = osp.join(dataset_dir, 'classes.txt')
    assert os.path.exists(class_filepath), f'please make sure the classes.txt is in path: {dataset_dir}'
    with open(class_filepath, 'r', encoding='utf-8') as f:
        classes = [str(x).strip() for x in f.readlines()]
    label_dir = osp.join(dataset_dir, label_dirname)
    image_dir = osp.join(dataset_dir, image_dirname)
    filenames = os.listdir(label_dir)
    label_files = [osp.join(label_dir, x) for x in filenames]
    save_dir = osp.join(dataset_dir, 'labels')
    os.makedirs(save_dir, exist_ok=True)

    for i, label_file in tqdm(enumerate(label_files), total=len(label_files)):
        out_file = osp.join(save_dir, Path(label_file).name)
        # 在图片文件夹中寻找相同文件名称的图片文件
        try:
            image_file = glob.glob(osp.join(image_dir, filenames[i].rsplit('.')[0]) + '.*')[0]
        except Exception as e:
            assert 0, f"please make sure the folder: {image_dir} have correct image and imagename"
        assert image_file.rsplit('.')[-1] in IMG_FORMATS, f'find file: {image_file}, but is not an image format'
        img_wh = Image.open(image_file).size
        dota2yolo(label_file, out_file, classes, img_wh)
    # print msg
    print(f"convert dataset: {Path(dataset_dir).name} dota label to yolo Success!!!")
    print(f"convert label is in folder labels")


if __name__ == '__main__':
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        image_dirname=opt.image_dirname,
        label_dirname=opt.label_dirname)