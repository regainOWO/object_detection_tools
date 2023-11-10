"""
此脚本的主要内容
1. 将fair1m的数据格式转换成dota格式
2. 标签名称中间如果有空格，则名称中间的空格字符则将替换成'-'符号；如 tennis racket -> tennis-racket

使用方法，需要输入三个参数：
--ann-dir: pascal voc的xml标签文件存放的路径
--output-dir: dota格式输出的路径，默认路径为pascal voc标签文件夹的同级 labelTxt目录
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
LOGGER = get_logger('fair1m2dota')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-dir', type=str, required=True,
                        help='input pascal voc annotations dir')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output yolo annotation dir, default is beside the ann-dir directory called labelTxt')
    parser.add_argument('--noempty', action='store_true',
                        help='if or not save empty label, default is not save')
    opt = parser.parse_args()
    return opt


def xml2dota_txt(xml_file, out_file, class_names, save_empty):
    """
    将fair1m的.xml标签文件转成 dota标签的.txt标签文件
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
    objects = root.find('objects')
    difficult = 0
    content_lines = []
    for obj in objects.iter('object'):
        is_pixel_point = True
        if obj.find('coordinate').text != 'pixel':
            is_pixel_point = False
        name = obj.find('possibleresult').find('name').text
        name = name.strip().replace(' ', '-')   # 将类别名称中间的空格字符替换成 -
        # 用于打印数据集中的总共的类别名称
        if name not in class_names:
            class_names.append(name)
        points = obj.find('points')
        poly = []
        for point in points.findall('point')[:4]:
            x, y = map(float, point.text.split(','))
            poly.append(x)
            poly.append(y)
        # 非像素坐标值，目前无法转化
        if not is_pixel_point:
            LOGGER.warning(f"file: {xml_file} coordinate is not in pixel")
            LOGGER.warning("not support geometry coordinate transfer to pixel coodinate yet! this label will be ignore")
            continue
        one_line = " ".join([str(a) for a in poly]) + " " + name + " " + str(difficult) + "\n"
        content_lines.append(one_line)
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
        xml2dota_txt(ann_file, output_file, class_names, save_empty)
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

# E:/GithubProjects/jp-sample/mini_fair1m/train/labelXml