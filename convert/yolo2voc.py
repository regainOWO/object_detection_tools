"""
此脚本的主要内容
1. 将yolo的数据格式转换成pascal voc格式，仅支持水平框格式

数据集目录结构
.
├── images
│   ├── ......
│   └── xxx.jpg
├── labels
│   ├── ......
│   └── xxx.txt
└── classes.txt


使用方法，需要输入三个参数：
--ann-dir: yolo的.txt标签文件存放的路径
--img-dir: 标签文件对应图片的存放路径
--classes: 标签的类别名称
--classes-file: 标签类别名称classes.txt文件的路径，和参数--classes选择一个输入即可
--output-dir: yolo格式输出的路径，默认路径为pascal voc标签文件夹的同级 annotations目录

"""

import argparse
import glob
import logging
from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import os.path as osp
from PIL import Image
from PIL.Image import init

from tqdm import tqdm

init()
from PIL.Image import EXTENSION


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
LOGGER = get_logger('yolo2voc')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-dir', type=str, required=True,
                        help='input yolo annotations dir')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='input yolo annotations dir')
    parser.add_argument('--classes', type=str, nargs='+',
                        help='input yolo class names')
    parser.add_argument('--classes-file', type=str, default=None,
                        help='classes.txt file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output pascal voc annotation dir, default is beside the ann-dir directory called Annotations')
    opt = parser.parse_args()
    return opt


def xywhn2xyxy(size, box):
    """反归一化，并转成所上角和右下角的两个坐标点"""
    x = box[0] * size[0]
    y = box[1] * size[1]
    w = box[2] * size[0]
    h = box[3] * size[1]
    xmin = int(x - w / 2.0)
    ymin = int(y - h / 2.0)
    xmax = int(x + w / 2.0)
    ymax = int(y + h / 2.0)
    return xmin, ymin, xmax, ymax


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
    width, height, channel = img_shape     # 获取图片的宽高

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


def txt2xml(txt_file, img_file, out_file, class_names):
    """
    将yolo标签的.txt标签文件转成pascal voc的.xml标签文件
    Args:
        txt_file (str): yolo标签文件的路径
        img_file (str): 图片文件的路径
        out_file (str): yolo标签文件的输出路径
        class_names (list): 标签类别名称
    """
    annotations = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = [x.strip() for x in f.readlines()]
    # 获取图片的尺寸信息
    img_shape = get_image_info(img_file)
    for line in lines:
        label = line.split(' ')
        label_idx = int(label[0])
        xywhn = list(map(float, label[1:5]))
        assert label_idx + 1 <= len(class_names), 'class_names length is not correct'
        name = class_names[label_idx]
        xyxy = xywhn2xyxy(img_shape[0:2], xywhn)
        annotation = {
            'name': name,
            'xmin': xyxy[0],
            'ymin': xyxy[1],
            'xmax': xyxy[2],
            'ymax': xyxy[3],
            'difficult': 0,
        }
        annotations.append(annotation)
    generate_voc_xml_file(out_file, img_file, img_shape, annotations)


def save_classes(classes_file, class_names):
    """保存classes.txt文件"""
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(class_names) + '\n')


def run(ann_dir, img_dir, class_names, output_dir):
    # 数据集所包含的标签类别名称
    if output_dir is None:
        output_dir = osp.join(osp.dirname(ann_dir), 'Annotations')
    # 创建生成label的文件夹
    os.makedirs(output_dir, exist_ok=True)
    # 后去所有的标签源文件
    ann_files = glob.glob(ann_dir + '/*.txt')
    assert len(ann_files) > 0, f"path: {ann_dir} dose not have any .txt file"
    preprocess_pbar = tqdm(ann_files, total=len(ann_files), desc='preprocess label and image file')
    # 过滤得到真实有效的样本（标签文件和图片文件都存在）
    valid_sample = []
    for ann_file in preprocess_pbar:
        filename = osp.basename(ann_file)
        prefix_name = osp.splitext(filename)[0]
        files = glob.glob(f"{img_dir}/{prefix_name}*")
        img_file = None
        # 过滤得到图片的路径
        for file in files:
            file_suffix = osp.splitext(file)[-1]
            if file_suffix in EXTENSION:
                img_file = file
                break
        if img_file is None:
            LOGGER.warning(f'lable file {ann_file} dose not have the same name image file in {img_dir}, this label will be ignore')
            continue
        valid_sample.append((img_file, ann_file))

    # 标签文件转化
    pbar = tqdm(valid_sample, total=len(ann_files))
    for img_file, ann_file in pbar:
        pbar.set_description(f'process {ann_file}')
        output_filename = osp.splitext(osp.basename(ann_file))[0] + '.xml'
        output_file = osp.join(output_dir, output_filename)
        txt2xml(ann_file, img_file, output_file, class_names)
    # save classes.txt
    classes_file = osp.join(osp.dirname(output_dir), 'classes.txt')
    save_classes(classes_file, class_names)
    # print msg
    LOGGER.info(f"dataset total class_names: {class_names}")
    LOGGER.info(f"classes.txt saved in: {classes_file}")
    LOGGER.info(f"pascal voc type label saved in: {output_dir}")


if __name__ == "__main__":
    opt = parse_opt()
    class_names = []
    if len(opt.classes) > 0:
        class_names = opt.classes
    elif opt.classes_file:
        assert osp.exists(opt.classes_file), f'{opt.classes_file} dose not exist'
        with open(opt.classes_file, 'r', encoding='utf-8') as f:
            class_names = [x.strip() for x in f.readlines()]
    else:
        assert 0, '请设置输入参数 --classes 后者 --classes-file'
    run(ann_dir=opt.ann_dir,
        img_dir=opt.img_dir,
        class_names=class_names,
        output_dir=opt.output_dir)