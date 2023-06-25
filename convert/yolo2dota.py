"""
此脚本的主要内容
1. 将yolo的数据格式转换成dota格式，支持水平框

数据集目录
.
├── images
│   ├── ...
│   └── **.jpg
└── labels
    ├── ...
    └── **.txt

使用方法，需要输入三个参数：
--dataset-dir: 数据集的根路径，数据集的跟路径下必须包含一个class.txt文件，里面包含了标签的类别信息
--image-dirname: 数据集根路径下存放所有图片的文件夹名称
--label-dirname: 数据集根路径下存放所有标签的文件夹名称
--output_path: 输出的文件夹路径
"""
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--image-dirname', type=str, default='images', help='image directory name under the dataset dir')
    parser.add_argument('--label-dirname', type=str, default='labels',
                        help='annotation directory name under the dataset dir')
    parser.add_argument('--output-path', type=str, default=None,
                        help='输出的文件夹路径，默认路径在dataset dir目录下的labelTxt文件夹')
    opt = parser.parse_args()
    return opt


def xyxy2dota(box):
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[1]
    x3, y3 = box[2], box[3]
    x4, y4 = box[0], box[3]
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def xywh2dota(box):
    cx, cy, w, h = box
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx - w // 2, cy + h // 2
    x3, y3 = cx + w // 2, cy + h // 2
    x4, y4 = cx + w // 2, cy - h // 2

    return [x1, y1, x2, y2, x3, y3, x4, y4]


def yolo2dota(yolo_label_file, img_width, img_height, out_file, class_names=None):

    with open(yolo_label_file, mode='r', encoding='utf-8') as f:
        contents = [x.strip() for x in f.readlines()]

    new_contents = []
    for content in contents:
        idx, x, y, w, h = content.split(' ')
        name = idx if class_names is None else class_names[int(idx)]
        box = [float(x) * img_width, float(y) * img_height, float(w) * img_width, float(h) * img_height]
        poly = xywh2dota(box)
        poly = ['%.1f' % x for x in poly]  # 转成字符串
        one_line = ' '.join(poly) + ' ' + name + ' ' + '0' + '\n'
        new_contents.append(one_line)
    # 保存dota格式的标签txt文件
    with open(out_file, mode='w', encoding='utf-8') as f:
        f.writelines(new_contents)


def run(dataset_dir, image_dirname, label_dirname, output_path):
    # 读取类别信息
    class_names_path = os.path.join(dataset_dir, 'classes.txt')
    class_names = None
    if os.path.isfile(class_names_path):
        with open(class_names_path, mode='r', encoding='utf-8') as f:
            class_names = [x.strip() for x in f.readlines()]
            if len(class_names) == 0:
                class_names = None
    image_dir = os.path.join(dataset_dir, image_dirname)
    label_dir = os.path.join(dataset_dir, label_dirname)
    label_files, image_files = [], []
    for filename in os.listdir(label_dir):
        file_basename, suffix = os.path.splitext(filename)
        if suffix != '.txt':
            continue
        label_file = os.path.join(label_dir, filename)
        try:
            image_file = glob.glob(os.path.join(image_dir, file_basename + '.*'))[0]
        except Exception as e:
            print(f"please make sure the folder: {image_dir} have correct image named like {file_basename}")
            continue
        label_files.append(label_file)
        image_files.append(image_file)

    assert len(label_files), f"{label_dir}下没有txt标签文件"
    if output_path is None or not isinstance(output_path, str):
        output_path = os.path.join(dataset_dir, 'labelTxt')
    # 创建生成label的文件夹
    os.makedirs(output_path, exist_ok=True)
    for image_file, label_file in tqdm(zip(image_files, label_files), total=len(label_files), desc=f'convert yolo label to dota label'):
        filename = Path(label_file).stem
        dst_file = os.path.join(output_path, filename + '.txt')  # 生成的文件
        image = Image.open(image_file)
        width, height = image.size
        yolo2dota(label_file, img_width=width, img_height=height, out_file=dst_file, class_names=class_names)
    print("convert yolo label to dota label success!")


if __name__ == "__main__":
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        image_dirname=opt.image_dirname,
        label_dirname=opt.label_dirname,
        output_path=opt.output_path)
