"""
此脚本的主要内容
1. 将voc的数据格式转换成dota格式，支持水平框
2. 并生成三个文件train.txt, val.txt, test.txt，里面的内容为图片的绝对路径。并不会对图片文件进行移动或拷贝，并不会产生重复的图片，从而过多占用硬盘空间。

数据集目录
.
├── Annotations
│   ├── ...
│   └── **.xml
├── ImageSets
│   └── Main
│       ├──test.txt
│       ├──train.txt
│       ├──trainval.txt
│       └──val.txt
└── JPEGImages
    ├── ...
    └── **.jpg

使用方法，需要输入三个参数：
--dataset-dir: 数据集的根路径
--image-dirname: 数据集根路径下存放所有图片的文件夹名称
--anno-dirname: 数据集根路径下存放所有xml标签的文件夹名称
"""

import logging
import xml.etree.ElementTree as ET
import os
from pathlib import Path

from tqdm import tqdm
import argparse

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
CLASS_NAMES = {}


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--image-dirname', type=str, default='JPEGImages', help='image directory name under the dataset dir')
    parser.add_argument('--anno-dirname', type=str, default='Annotations',
                        help='annotation directory name under the dataset dir')
    opt = parser.parse_args()
    return opt


def xyxy2dota(box):
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[1]
    x3, y3 = box[2], box[3]
    x4, y4 = box[0], box[3]
    return x1, y1, x2, y2, x3, y3, x4, y4


def xml2txt(xml_file, out_file):
    """
    @dataset_dir: 数据集存放annotation xml文件的目录
    @image_id: 文件名称

    return: img suffix 返回图片文件的后缀
    """
    global CLASS_NAMES
    in_file = open(xml_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    image_filename = Path(root.find('filename').text)   # maybe unvaluable, just use the suffix
    image_suffix = os.path.splitext(image_filename)[-1].split('.')[-1]
    assert image_suffix in IMG_FORMATS, f'{image_filename} is not image type, please check your dataset'
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    content_lines = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # 用于打印数据集中的总共的类别名称
        if cls not in CLASS_NAMES.keys():
            CLASS_NAMES[cls] = 1
        else:
            CLASS_NAMES[cls] += 1
        xmlbox = obj.find('bndbox')
        if xmlbox:
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(
                xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            bbox = xyxy2dota(b)
        else:
            logging.info(f'file: {in_file} content object do not have bndbox, this object will be ignore')
            continue
        content_lines.append(" ".join([str(a) for a in bbox]) + " " + cls + " " + difficult + '\n')
    if len(content_lines) == 0:
        logging.warning(f'no lines to save, the file: {out_file} is empty')
    # 将内容写入文件
    with open(out_file, 'w', encoding='utf-8') as f:
        f.writelines(content_lines)
    return image_suffix


def get_sets(set_dir):
    """
    去除不存在txt文件的 set
    """
    sets = ['train', 'val', 'test']
    filenames = os.listdir(set_dir)
    for i, set in enumerate(sets):
        set_file = set + '.txt'
        if set_file not in filenames:
            sets.pop(i)
    assert len(sets) > 2 or 'train' in sets, f'please make sure your {set_dir} path have train.txt, val.txt(Optional), test.txt(Optional), you can use split_data.py script to split data'
    return sets


def run(dataset_dir, image_dirname, anno_dirname):
    # 读取类别信息
    # 读取文件
    image_sets_dir = os.path.join(dataset_dir, 'ImageSets', 'Main')
    sets = get_sets(image_sets_dir)
    # 创建生成label的文件夹
    label_dirname = 'labelTxt'
    label_dir = os.path.join(dataset_dir, label_dirname)
    os.makedirs(label_dir, exist_ok=True)
    for image_set in sets:
        # 读取图片文件名称
        with open(f'{image_sets_dir}/{image_set}.txt') as f:
            image_ids = [str(x).strip() for x in f.readlines()]
        # 生成新的包含图片路径的文件，存放到数据集的根目录
        list_file = open(f'{dataset_dir}/{image_set}.txt', 'w', encoding='utf-8')
        images_dir = f'{dataset_dir}/{image_dirname}'
        for image_id in tqdm(image_ids, desc=f'convert {image_set} label'):
            # 将xml文件转成txt
            xml_file = os.path.join(dataset_dir, anno_dirname, f'{image_id}.xml')
            out_file = os.path.join(label_dir, f'{image_id}.txt')
            image_suffix = xml2txt(xml_file, out_file)
            # 写入图片文件路径
            list_file.write(f'{images_dir}/{image_id}.{image_suffix}\n')
        list_file.close()
    # print msg
    print(f"dataset xml file total class name count: {CLASS_NAMES}")
    print(f"convert dataset: {Path(dataset_dir).name} voc label to yolo Success!!!")
    print(f"in folder: {dataset_dir}. generate train.txt、val.txt and test.txt Success!!!")
    print(f"convert label is in folder {label_dirname}")


if __name__ == "__main__":
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        image_dirname=opt.image_dirname,
        anno_dirname=opt.anno_dirname)
