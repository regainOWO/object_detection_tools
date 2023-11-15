"""
此脚本的主要内容:
将coco的数据集格式转换成pascal voc格式
coco2017数据集目录
.
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017
│   ├── ...
│   └── **.jpg
└── val2017
    ├── ...
    └── **.jpg

使用方法，需要输入四个参数：
--ann-file: coco标签文件的路径。
--output-dir: pascal voc标签文件输出的路径，默认值为参数ann-file父级目录同级的Annotations目录

"""
import argparse
import logging
from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import os.path as osp
import time
from tqdm import tqdm
from pathlib import Path

try:
    import orjson as json
except ModuleNotFoundError:
    print("install orjson package makes read json file more quickly! ---->  \033[91mpip install orjson\033[0m")
    import json


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
    parser.add_argument('--ann-file', type=str, required=True, help='input coco .json label file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='the pascal voc Annotations dir, default is beside the parent of the ann-file')
    opt = parser.parse_args()
    return opt


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
    r_folder.text = "Unknown"

    r_filename = ET.SubElement(root, 'filename')
    r_filename.text = osp.basename(image_file)
    r_path = ET.SubElement(root, 'path')
    r_path.text = "Unknown"

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



def coco2voc(ann_file, output_dir):
    filename = osp.splitext(osp.basename(ann_file))[0]
    LOGGER.info(f"开始读取 {osp.abspath(ann_file)} ......")
    t1 = time.time()
    with open(ann_file, 'r') as f:
        data = json.loads(f.read())
    t2 = time.time() - t1
    LOGGER.info(f"cost: {t2}")

    id_map = {}
    names = {}

    # 解析目标类别，也就是 categories 字段，并将类别写入文件 classes.txt 中，存放在label_dir的同级目录中
    output_dir_parent = Path(output_dir).parent.as_posix()
    with open(osp.join(output_dir_parent, f'{filename}_classes.txt'), 'w', encoding='utf-8') as f:
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
            names[i] = category['name']
    LOGGER.info(f"generate classes.txt under the {output_dir_parent} Success!!")

    annotations = {}
    for ann in tqdm(data['annotations'], desc=f'preprocessing the label file annotations: {ann_file}'):
        img_id = ann['image_id']
        xmin, ymin, width, height = ann["bbox"]
        xmax = xmin + width
        ymax = ymin + height
        one_annotation = {
            'name': names[id_map[ann["category_id"]]],
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'difficult': 0
        }
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(one_annotation)
    # 开始保存pascal voc xml file
    for img in tqdm(data['images'], total=len(data['images']), desc="convert data...."):

        # 解析 images 字段，分别取出图片文件名、图片的宽和高、图片id
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]

        # label文件名，与对应图片名只有后缀名不一样
        label_filename = osp.splitext(filename)[0] + ".xml"
        label_file = osp.join(output_dir, label_filename)
        annos = annotations.get(img_id, [])
        # 保存成pascal voc 的标签格式
        generate_voc_xml_file(label_file, filename, (img_width, img_height, 3), annos)
        
    

def run(ann_file, output_dir):
    assert osp.exists(ann_file), f"{ann_file} not exists"
    if output_dir is None:
        output_dir = Path(ann_file).parent.parent.joinpath('Annotations').as_posix()
    os.makedirs(output_dir, exist_ok=True)
    coco2voc(ann_file, output_dir)


if __name__ == '__main__':
    opt = parse_opt()
    run(ann_file=opt.ann_file,
        output_dir=opt.output_dir)
