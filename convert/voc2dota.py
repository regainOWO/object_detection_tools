"""
此脚本的主要内容
1. 将voc的数据格式转换成dota格式，支持水平框
2. voc的标签名称中间如果有空格，则名称中间的空格字符则将替换成'-'符号；如 tennis racket -> tennis-racket

使用方法，需要输入三个参数：
--ann-dir: pascal voc的xml标签文件存放的路径
--output-dir: dota格式输出的路径
--noempty: 是否保存空标签文件，默认是保存的
"""

import argparse
import glob
import logging
import xml.etree.ElementTree as ET
import os
import os.path as osp

from tqdm import tqdm


def get_logger(log_name):
    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler()
    ch_format = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s')
    ch.setFormatter(ch_format)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger


# logger
LOGGER = get_logger('voc2dota')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-dir', type=str, required=True,
                        help='input pascal voc annotations dir')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output dota annotation dir, default is beside the ann-dir directory called labels')
    parser.add_argument('--noempty', action='store_true',
                        help='if or not save empty label, default is not save')
    opt = parser.parse_args()
    return opt


def xyxy2dota(box):
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[1]
    x3, y3 = box[2], box[3]
    x4, y4 = box[0], box[3]
    return x1, y1, x2, y2, x3, y3, x4, y4


def xml2txt(xml_file, out_file, class_names, save_empty):
    """
    将pascal voc的.xml标签文件转成 dota标签的.txt标签文件
    Args:
        xml_file (str): pascal voc标签文件的路径
        out_file (str): dota标签文件的输出路径
        class_names (list): 标签类别名称
        save_empty (bool): 是否保存空标签
    """
    in_file = open(xml_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    content_lines = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        name = obj.find('name').text
        name.replace(' ', '-')  # 将类别名称的空格间隙替换成 '-'
        # 用于打印数据集中的总共的类别名称
        if name not in class_names:
            class_names.append(name)
        xmlbox = obj.find('bndbox')
        if xmlbox:
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(
                xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            bbox = xyxy2dota(b)
        else:
            LOGGER.info(f'file: {in_file} content object do not have bndbox, this object will be ignore')
            continue
        content_lines.append(" ".join([str(a) for a in bbox]) + " " + name + " " + difficult + '\n')
    if len(content_lines) == 0:
        LOGGER.warning(f'no lines to save, the file: {out_file} is empty')
    if len(content_lines) > 0 or save_empty:
        # 将内容写入文件
        with open(out_file, 'w', encoding='utf-8') as f:
            f.writelines(content_lines)


def save_classes(classes_file, class_names):
    """保存classes.txt文件"""
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(class_names) + '\n')


def run(ann_dir, output_dir, save_empty=True):
    # 数据集所包含的标签类别名称
    class_names = []
    if output_dir is None:
        output_dir = osp.join(osp.dirname(ann_dir), 'labelTxt')
    # 创建生成label的文件夹
    os.makedirs(output_dir, exist_ok=True)
    # 后去所有的标签源文件
    ann_files = glob.glob(ann_dir + '/*.xml')
    assert len(ann_files) > 0, f"path: {ann_dir} dose not have any .xml file"
    # 标签文件转化
    pbar = tqdm(ann_files, total=len(ann_files))
    for ann_file in pbar:
        pbar.set_description(f'process {ann_file}')
        output_filename = osp.splitext(osp.basename(ann_file))[0] + '.txt'
        output_file = osp.join(output_dir, output_filename)
        xml2txt(ann_file, output_file, class_names, save_empty)
    # save classes.txt
    classes_file = osp.join(osp.dirname(output_dir), 'classes.txt')
    class_names.sort()
    save_classes(classes_file, class_names)
    # print msg
    LOGGER.info(f"dataset total class_names: {class_names}")
    LOGGER.info(f"classes.txt saved in: {classes_file}")
    LOGGER.info(f"dota type label saved in: {output_dir}")


if __name__ == "__main__":
    opt = parse_opt()
    run(ann_dir=opt.ann_dir,
        output_dir=opt.output_dir,
        save_empty=not opt.noempty)
