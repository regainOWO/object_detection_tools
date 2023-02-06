"""
此脚本的主要内容
1. 将voc的数据格式转换成coco格式，支持水平框和旋转框
2. 并生成三个文件train.txt, val.txt, test.txt，里面的内容为图片的绝对路径。并不会对图片文件进行移动或拷贝，并不会产生重复的图片，从而过多占用硬盘空间。

使用方法，需要输入三个参数：
--dataset-dir: 数据集的根路径
--obb: 是否提取标签文件中旋转框的标签，默认不开启，即只提取水平框的标签

xml的标签文件路径：必须在数据集根文件夹下的Annotations文件夹里
输入的路径：根据参数obb决定，若obb不开启，则文件夹名称为labels，若开启则文件夹名称为labels_obb (放在数据集的根文件夹中）
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
    parser.add_argument('--obb', action='store_true', help='convert obb label, default convert hbb label')
    opt = parser.parse_args()
    return opt


# xyxy2xywh 并归一化
def xyxy2xywh_normalize(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


# 弧度转成角度 并归一化
def xywha_normalize(size, rbox):
    pi = 3.141592653589793
    dw = 1./(size[0])
    dh = 1./(size[1])
    x, y, w, h, a = rbox
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    a = int(a * 180 / pi)   # 弧度转成角度
    return (x, y, w, h, a)


def xml2txt(xml_file, out_file, classes, isOBB=False):
    """
    @dataset_dir: 数据集存放annotation xml文件的目录
    @image_id: 文件名称
    @classes: 检测类别名称

    return: img suffix 返回图片文件的后缀
    """
    global CLASS_NAMES
    in_file = open(xml_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    image_filepath = Path(root.find('path').text).as_posix()   # maybe unvaluable, just use the suffix
    image_suffix = image_filepath.split('.')[-1].lower()
    assert image_suffix in IMG_FORMATS, f'{image_filepath.name} is not image type, please check your dataset'
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    content_lines = []
    for obj in root.iter('object'):
        #difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # 用于打印数据集中的总共的类别名称
        if cls not in CLASS_NAMES.keys():
            CLASS_NAMES[cls] = 1
        else:
            CLASS_NAMES[cls] += 1
        # 若cls不在筛选的classes里，则跳过
        if cls not in classes:
            continue
        cls_id = classes.index(cls)     # 类别索引
        object_type = obj.find('type').text
        xmlbox = obj.find(object_type)
        if isOBB and object_type == 'robndbox':
            assert object_type == 'robndbox', f'check file: {in_file} content, cannot find robndbox, maybe is hbb, you do not set obb'
            b = (float(xmlbox.find('cx').text), float(xmlbox.find('cy').text), float(
                xmlbox.find('w').text), float(xmlbox.find('h').text), float(xmlbox.find('angle').text))
            bbox = xywha_normalize((w, h), b)
        elif not isOBB and object_type == 'bndbox':
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
                xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bbox = xyxy2xywh_normalize((w, h), b)
        else:
            logging.info(f'file: {in_file} content have {object_type} type, you set isOBB is {isOBB}, this type will be ignore')
            continue
        content_lines.append(str(cls_id) + " " + " ".join([str(a) for a in bbox]) + '\n')
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


def run(dataset_dir, isOBB=False):
    # 读取类别信息
    class_filepath = os.path.join(dataset_dir, 'classes.txt')
    assert os.path.exists(class_filepath), f'please make sure the classes.txt is in path: {dataset_dir}'
    with open(class_filepath) as f:
        classes = [str(x).strip() for x in f.readlines()]
    # 读取文件
    image_sets_dir = os.path.join(dataset_dir, 'ImageSets', 'Main')
    sets = get_sets(image_sets_dir)
    # 创建生成label的文件夹
    label_dirname = 'labels' if not isOBB else 'labels_obb'
    label_dir = os.path.join(dataset_dir, label_dirname)
    os.makedirs(label_dir, exist_ok=True)
    for image_set in sets:
        # 读取图片文件名称
        with open(f'{image_sets_dir}/{image_set}.txt') as f:
            image_ids = [str(x).strip() for x in f.readlines()]
        # 生成新的包含图片路径的文件，存放到数据集的根目录
        list_file = open(f'{dataset_dir}/{image_set}.txt', 'w', encoding='utf-8')
        images_dir = f'{dataset_dir}/JPEGImages'
        for image_id in tqdm(image_ids, desc=f'convert {image_set} label'):
            # 将xml文件转成txt
            xml_file = os.path.join(dataset_dir, 'Annotations', f'{image_id}.xml')
            out_file = os.path.join(label_dir, f'{image_id}.txt')
            image_suffix = xml2txt(xml_file, out_file, classes, isOBB=isOBB)
            # 写入图片文件路径
            list_file.write(f'{images_dir}/{image_id}.{image_suffix}\n')
        list_file.close()
    print(CLASS_NAMES)
    return label_dirname


if __name__ == "__main__":
    opt = parse_opt()
    out_label_dirname = run(dataset_dir=opt.dataset_dir,
                            isOBB=opt.obb)
    print(f"convert dataset: {Path(opt.dataset_dir).name} voc label to yolo Success!!!")
    print(f"in folder: {opt.dataset_dir}. generate train.txt、val.txt and test.txt Success!!!")
    print(f"convert label is in folder {out_label_dirname}")