"""
此脚本的主要内容
* 将fair1m的数据格式转换成dota格式
* points的数据默认是像素坐标，还未支持地理坐标转像素坐标

数据集目录
.
├── images
│   ├──xxx.tif
│   └──......
└── labelXml
    ├──xxx.xml
    └──......

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
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='input dataset dir')
    parser.add_argument('--image-dirname', type=str, default='images',
                        help='image directory name under the dataset dir')
    parser.add_argument('--label-dirname', type=str, default='labelXml',
                        help='annotation directory name under the dataset dir')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output directory path, default is labelTxt directory under the dataset dir')
    opt = parser.parse_args()
    return opt


def xml2dota_txt(image_dir, xml_file, out_file):
    """
    Args:
        image_dir (str): 图片文件夹的路径
        xml_file (str): xml标签文件的路径
        out_file (str): 输出标签文件的路径
    """
    global CLASS_NAMES
    in_file = open(xml_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    source = root.find('source')
    image_filename = Path(source.find('filename').text)  # maybe unvaluable, just use the suffix
    image_suffix = os.path.splitext(image_filename)[-1].split('.')[-1]
    assert image_suffix in IMG_FORMATS, f'{image_filename} is not image type, please check your dataset'
    image_file = os.path.join(image_dir, image_filename)
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
        class_name = obj.find('possibleresult').find('name').text
        class_name = class_name.strip().replace(' ', '-')   # 将类别名称中间的空格字符替换成 -
        # 用于打印数据集中的总共的类别名称
        if class_name not in CLASS_NAMES.keys():
            CLASS_NAMES[class_name] = 1
        else:
            CLASS_NAMES[class_name] += 1
        points = obj.find('points')
        poly = []
        for point in points.findall('point')[:4]:
            x, y = map(float, point.text.split(','))
            if not is_pixel_point:
                logging.warning('')
            poly.append(x)
            poly.append(y)
        one_line = " ".join([str(a) for a in poly]) + " " + class_name + " " + str(difficult) + "\n"
        content_lines.append(one_line)
    if len(content_lines) == 0:
        logging.warning(f'no lines to save, the file: {out_file} is empty')
    # 将内容写入文件
    with open(out_file, 'w', encoding='utf-8') as f:
        f.writelines(content_lines)


def run(dataset_dir, image_dirname, label_dirname, output_dir):
    # 读取类别信息
    image_dir = os.path.join(dataset_dir, image_dirname)
    label_dir = os.path.join(dataset_dir, label_dirname)
    label_files = []
    for label_filename in os.listdir(label_dir):
        if os.path.splitext(label_filename)[-1] != '.xml':
            continue
        label_file = os.path.join(label_dir, label_filename)
        label_files.append(label_file)
    assert len(label_files), f"{label_dir}下没有xml标签文件"
    if output_dir is None or not isinstance(output_dir, str):
        output_dir = os.path.join(dataset_dir, 'labelTxt')
    # 创建生成label的文件夹
    os.makedirs(output_dir, exist_ok=True)
    for label_file in tqdm(label_files, total=len(label_files), desc=f'convert fair1m label to dota label'):
        filename = Path(label_file).stem
        dst_file = os.path.join(output_dir, filename + '.txt')  # 生成的文件
        xml2dota_txt(image_dir, label_file, dst_file)
    logging.info("convert fair1m label to dota label success!")
    # 生成classes.txt文件
    class_names = list(CLASS_NAMES.keys())
    class_names.sort()
    with open(os.path.join(dataset_dir, 'classes.txt'), mode='w', encoding='utf-8') as f:
        content_lines = [x + '\n' for x in class_names]
        f.writelines(content_lines)
    logging.info(f"Success generate classes.txt in {os.path.abspath(dataset_dir)}")


if __name__ == "__main__":
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        image_dirname=opt.image_dirname,
        label_dirname=opt.label_dirname,
        output_dir=opt.output_dir)
