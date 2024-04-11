"""
此脚本的主要内容
将labelme的数据格式转换成pascal voc格式

使用方法，需要输入三个参数：
--ann-dir: labelme的json标签文件存放的路径
--output-dir: pascal voc格式输出的路径，默认路径为labelme标签文件夹的同级 Annotations目录
--noempty: 是否保存空标签文件，默认是保存的

1148 = 988 + 160  392 450 538
"""

import argparse
import glob
import json
import logging
import shutil
from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import os.path as osp
from PIL import Image

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
                        help='output dota annotation dir, default is beside the ann-dir directory called labelTxt')
    parser.add_argument('--noempty', action='store_true',
                        help='if or not save empty label, default is not save')
    opt = parser.parse_args()
    return opt


def get_image_info(image_file):
    """
    获取图片的宽、高、通道数
    Args:
        image_file (str): 图片路径
    Returns:
        width (int): 宽
        height (int): 高
        channel (int): 通道数
    """
    image = Image.open(image_file)
    width, height = image.size
    channel = len(image.getbands())
    return width, height, channel


def format_save_xml_file(root, xml_file, indent="\t", newl="\n", encoding="utf-8"):
    """
    格式化输出xml文件
    reference: https://blog.csdn.net/qq_29007291/article/details/106666001
    Args:
        root (Element):
        xml_file (str): xml文件路径
    """
    raw_text = ET.tostring(root)
    dom = minidom.parseString(raw_text)
    with open(xml_file, 'w', encoding=encoding) as f:
        dom.writexml(f, "", indent, newl, encoding)


def str_list_to_txtfile(txt_file: str, data: list):
    """
    将字符列表保存成.txt文件
    Args:
        txt_file (str): .txt文件路径
        data (list): 字符数组
    """
    with open(txt_file, 'w', encoding='utf-8') as f:
        contents = [x + '\n' for x in data]
        f.writelines(contents)


def generate_voc_xml_file(xml_file, image_file, img_shape, annotations: list):
    """
    保存成pascal voc 格式.xml文件
    Args:
        xml_file (str): 标签保存路径
        image_file (str): 图片路径
        img_shape (str): 图片宽高和维度
        annotations (list): 标签内容
    """
    image_file = image_file.replace('\\', '/')
    height, width, channel = img_shape     # 获取图片的宽高

    root = ET.Element('annotation')
    r_folder = ET.SubElement(root, 'folder')
    r_folder.text = osp.dirname(image_file)

    r_filename = ET.SubElement(root, 'filename')
    r_filename.text = os.path.basename(image_file)
    r_path = ET.SubElement(root, 'path')
    r_path.text = image_file

    r_source = ET.SubElement(root, 'source')
    rr_database = ET.SubElement(r_source, 'database')
    rr_database.text = 'Unknown'

    r_size = ET.SubElement(root, 'size')
    rr_width = ET.SubElement(r_size, 'width')
    rr_width.text = str(width)
    rr_height = ET.SubElement(r_size, 'height')
    rr_height.text = str(height)
    rr_depth = ET.SubElement(r_size, 'depth')
    rr_depth.text = str(channel)

    r_segmented = ET.SubElement(root, 'segmented')
    r_segmented.text = str(0)

    for ann in annotations:
        r_obj = ET.SubElement(root, 'object')

        rr_name = ET.SubElement(r_obj, 'name')
        rr_name.text = str(ann['name'])
        rr_pose = ET.SubElement(r_obj, 'pose')
        rr_pose.text = 'Unspecified'
        rr_truncated = ET.SubElement(r_obj, 'truncated')
        rr_truncated.text = str(0)
        rr_difficult = ET.SubElement(r_obj, 'difficult')
        rr_difficult.text = str(ann['difficult'])

        rr_bndbox = ET.SubElement(r_obj, 'bndbox')
        rrr_xmin = ET.SubElement(rr_bndbox, 'xmin')
        rrr_ymin = ET.SubElement(rr_bndbox, 'ymin')
        rrr_xmax = ET.SubElement(rr_bndbox, 'xmax')
        rrr_ymax = ET.SubElement(rr_bndbox, 'ymax')
        rrr_xmin.text = str(ann['xmin'])
        rrr_ymin.text = str(ann['ymin'])
        rrr_xmax.text = str(ann['xmax'])
        rrr_ymax.text = str(ann['ymax'])

    format_save_xml_file(root, xml_file)


def labelme2xml(json_file, out_file, class_names, save_empty: bool):
    """
    将yolo标签的.txt标签文件转成pascal voc的.xml标签文件
    Args:
        txt_file (str): yolo标签文件的路径
        out_file (str): yolo标签文件的输出路径
        class_names (list): 标签类别名称
        save_empty (bool): 是否保存空标签
    """
    annotations = []
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    width, height = data['imageWidth'], data['imageHeight']
    img_shape = (height, width, 3)
    # img_file = osp.join(osp.dirname(json_file), data['imagePath'])
    # 获取图片的尺寸信息
    # img_shape = get_image_info(img_file)
    img_file = osp.basename(data['imagePath'])
    for shape in data['shapes']:
        name = shape['label']
        points = shape['points']
        annotation = {
            'name': name,
            'xmin': min(points[0][0], points[1][0]),
            'ymin': min(points[0][1], points[1][1]),
            'xmax': max(points[0][0], points[1][0]),
            'ymax': max(points[0][1], points[1][1]),
            'difficult': 0,
        }
        annotations.append(annotation)
        if name not in class_names:
            class_names.append(name)
    if len(annotations) > 0 or save_empty:
        generate_voc_xml_file(out_file, img_file, img_shape, annotations)
    else:
        filename = osp.basename(json_file)
        image_filename = osp.splitext(filename)[0] + '.jpg'
        image_file = osp.join(osp.dirname(osp.dirname(json_file)), 'images', image_filename)
        os.remove(image_file)
        LOGGER.warning(f"label file is empty: {json_file}")


def save_classes(classes_file, class_names):
    """保存classes.txt文件"""
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(class_names) + '\n')


def run(ann_dir, output_dir, save_empty=True):
    # 数据集所包含的标签类别名称
    class_names = []
    if output_dir is None:
        output_dir = osp.join(osp.dirname(ann_dir), 'Annotations')
    # 创建生成label的文件夹
    os.makedirs(output_dir, exist_ok=True)
    # 后去所有的标签源文件
    ann_files = glob.glob(ann_dir + '/*.json')
    assert len(ann_files) > 0, f"path: {ann_dir} dose not have any .json file"
    # 标签文件转化
    pbar = tqdm(ann_files, total=len(ann_files))
    for ann_file in pbar:
        pbar.set_description(f'process {ann_file}')
        output_filename = osp.splitext(osp.basename(ann_file))[0] + '.xml'
        output_file = osp.join(output_dir, output_filename)
        labelme2xml(ann_file, output_file, class_names, save_empty)
    # save classes.txt
    classes_file = osp.join(osp.dirname(output_dir), 'classes.txt')
    class_names.sort()
    save_classes(classes_file, class_names)
    # print msg
    LOGGER.info(f"dataset total class_names: {class_names}")
    LOGGER.info(f"classes.txt saved in: {classes_file}")
    LOGGER.info(f"voc type label saved in: {output_dir}")


if __name__ == "__main__":
    opt = parse_opt()
    run(ann_dir=opt.ann_dir,
        output_dir=opt.output_dir,
        save_empty=not opt.noempty)