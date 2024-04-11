import argparse
import glob
import json
import logging
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
                        help='input pascal voc Annotations dir')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='input pascal voc JPEGImage dir')
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


def generate_json_file(json_file, image_file, annotations: list):
    """
    保存成pascal voc 格式.xml文件
    Args:
        xml_file (str): 标签保存路径
        image_file (str): 图片路径
        img_shape (str): 图片宽高和维度
        annotations (list): 标签内容
    """
    width, height, _ = get_image_info(image_file)
    labelme_json = {
        "flags": {},
        "shapes": [],
        "imagePath": osp.relpath(image_file, osp.dirname(json_file)),
        "imageData": None,
        "imageHeight": width,
        "imageWidth": height
    }

    for annotation in annotations:
        shape = {
            "label": annotation['name'],
            "points": [
                [annotation['xmin'], annotation['ymin']],
                [annotation['xmax'], annotation['ymax']]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        }
        labelme_json['shapes'].append(shape)
    with open(json_file, mode='w', encoding='utf-8') as f:
        json.dump(labelme_json, f, ensure_ascii=False, indent=4)


def parse_xml(xml_file, class_names):
    in_file = open(xml_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    annotations = []
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
            annotation = {
                'name': name,
                'xmin': b[0],
                'ymin': b[1],
                'xmax': b[2],
                'ymax': b[3]
            }
            annotations.append(annotation)
    return annotations
    

def run(ann_dir: str, img_dir: str, output_dir: str = None):
    if output_dir is None:
        output_dir = osp.join(osp.dirname(ann_dir), 'labelme_Annotations')
    os.makedirs(output_dir, exist_ok=True)
    class_names = []
    xml_files = glob.glob(ann_dir + '/*.xml')
    for xml_file in tqdm(xml_files, total=len(xml_files)):
        filename = osp.basename(xml_file)
        filename_prefix = osp.splitext(filename)[0]
        json_file = osp.join(output_dir, filename_prefix + '.json')
        image_file = osp.join(img_dir, filename_prefix + '.jpg')
        annotations = parse_xml(xml_file=xml_file,
                                class_names=class_names)
        generate_json_file(json_file=json_file,
                           image_file=image_file,
                           annotations=annotations)
    print(class_names)

if __name__ == '__main__':
    opt = parse_opt()
    run(ann_dir=opt.ann_dir,
        img_dir=opt.img_dir,
        output_dir=opt.output_dir)
