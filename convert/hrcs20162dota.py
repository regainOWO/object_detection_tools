"""
此脚本的主要内容
1. 将HRCS-2016的数据格式转换成dota格式
2. 标签名称全转成ship

使用方法，需要输入三个参数：
--ann-dir: HRCS-2016的xml标签文件存放的路径
--output-dir: dota格式输出的路径，默认路径为HRCS-2016标签文件夹的同级 labelTxt目录
--noempty: 是否保存空标签文件，默认是保存的
"""

import argparse
import glob
import logging
import math
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
LOGGER = get_logger('hrcs20162dota')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-dir', type=str, required=True,
                        help='input HRCS-2016 annotations dir')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output dota annotation dir, default is beside the ann-dir directory called labelTxt')
    parser.add_argument('--noempty', action='store_true',
                        help='if or not save empty label, default is not save')
    opt = parser.parse_args()
    return opt


def xywha2poly(box):
    # reference: https://aistudio.baidu.com/projectdetail/3477659
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0]
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]
    return convert_box


def xml2txt(xml_file, out_file, class_names, save_empty):
    """
    将HRCS-2016的.xml标签文件转成 dota标签的.txt标签文件
    Args:
        xml_file (str): HRCS-2016标签文件的路径
        out_file (str): dota标签文件的输出路径
        class_names (list): 标签类别名称
        save_empty (bool): 是否保存空标签
    """
    in_file = open(xml_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    w = int(root.find('Img_SizeWidth').text)
    h = int(root.find('Img_SizeHeight').text)
    objects = root.find('HRSC_Objects')
    content_lines = []
    for obj in objects.iter('HRSC_Object'):
        difficult = '0'
        # name = obj.find('Class_ID').text
        name = 'ship'
        name.replace(' ', '-')  # 将类别名称的空格间隙替换成 '-'
        # 用于打印数据集中的总共的类别名称
        if name not in class_names:
            class_names.append(name)
        b = (float(obj.find('mbox_cx').text), float(obj.find('mbox_cy').text), float(
            obj.find('mbox_w').text), float(obj.find('mbox_h').text), float(obj.find('mbox_ang').text))
        bbox = xywha2poly(b)
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

# D:/迅雷下载/HRSC2016_dataset/HRCS-2016/Annotations