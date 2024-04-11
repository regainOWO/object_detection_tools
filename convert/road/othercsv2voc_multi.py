"""
此脚本的主要内容
1. 将外部框架的数据格式转换成pascal voc格式，仅支持水平框格式

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
--csv-file: .csv文件存放的路径
--img-dir: 标签文件对应图片的存放路径
--output-dir: yolo格式输出的路径，默认路径为pascal voc标签文件夹的同级 annotations目录

"""

import argparse
import glob
import logging
import shutil
import time
from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import os.path as osp
from PIL import Image
from PIL.Image import init
import pandas as pd

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
LOGGER = get_logger('csv2voc')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projection-root', type=str, required=True,
                        help='input .csv file path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output pascal voc annotation dir, default is beside the ann-dir directory called Annotations')
    opt = parser.parse_args()
    return opt


def is_image_file(image_file):
    """判断文件路径是否存在，且是否是图片后缀的格式"""
    if not osp.isfile(image_file):
        return False
    _, suffix = osp.splitext(image_file)
    if suffix in EXTENSION:
        return True
    return False


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


def generate_voc_xml_file(xml_file, image_file, annotations: list):
    """
    保存成pascal voc 格式.xml文件
    Args:
        xml_file (str): 标签保存路径
        image_file (str): 图片路径
        img_shape (str): 图片宽高和维度
        annotations (list): 标签内容
    """
    image_file = image_file.replace('\\', '/')
    width, height, channel = get_image_info(image_file)     # 获取图片的宽高

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


def save_classes(classes_file, class_names):
    """保存classes.txt文件"""
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(class_names) + '\n')

    

def preprocess_csv_file(csv_file, img_dir, class_names, separator=' '):
    df = pd.read_csv(csv_file)

    colunm_names = df.columns.tolist()
    data_info = {}
    process_csv_pbar = tqdm(df.iterrows(), desc=f"preprocess with the file: {csv_file}")
    for index, row in process_csv_pbar:
        object_name = row['key_object_category']
        bboxs1 = eval(row['bboxs1'])
        # print('bboxs1', bboxs1)
        # print('type', type(bboxs1))
        bboxs2 = eval(row['bboxs2'])
        # print('bboxs2', bboxs2)
        # print('type', type(bboxs2))
        # left
        filename_1 = row['image_name_1_close_shot']
        frame_name = str(filename_1).split('-')[-1]
        img_1 = osp.join(img_dir, filename_1)
        
        annotation_1 = {
            'name': object_name,
            'xmin': bboxs1[0],
            'ymin': bboxs1[1],
            'xmax': bboxs1[2],
            'ymax': bboxs1[3],
            'difficult': 0 if bboxs1[4] > 0.5 else 1,
        }
        data_info[img_1] = [annotation_1]
        # right
        # filename_2 = str(filename_1).replace('-1-', '-2-')
        # img_2 = osp.join(img_dir, filename_2)
        img_finds = glob.glob(img_dir + f'/*-{frame_name}')
        img_finds.remove(img_1)
        img_2 = img_finds[0]
        annotation_2 = {
            'name': object_name,
            'xmin': bboxs2[0],
            'ymin': bboxs2[1],
            'xmax': bboxs2[2],
            'ymax': bboxs2[3],
            'difficult': 0 if bboxs2[4] > 0.5 else 1,
        }
        data_info[img_2] = [annotation_2]
        if object_name not in class_names:
            class_names.append(object_name)
    return data_info


def copy_img(src_dir, dst_dir):
    for filename in os.listdir(src_dir):
        src_file = osp.join(src_dir, filename)
        if not is_image_file(src_file):
            continue
        dst_file = osp.join(dst_dir, filename)
        shutil.copy(src_file, dst_file)


def run(csv_file, img_dir, output_dir):
    class_names = []
    # 数据集所包含的标签类别名称
    if output_dir is None:
        output_dir = osp.join(osp.dirname(img_dir), 'Annotations')
    # 创建生成label的文件夹
    os.makedirs(output_dir, exist_ok=True)
    # 后去所有的标签源文件
    assert osp.isfile(csv_file) > 0, f"path: {csv_file} dose not exist"
    # 处理csv文件
    data_info = preprocess_csv_file(csv_file, img_dir, class_names)

    pbar = tqdm(data_info.items(), total=len(data_info), desc='save xml: ')
    # 标签文件转化
    for img_file, annotations in pbar:
        output_filename = osp.splitext(osp.basename(img_file))[0] + '.xml'
        output_file = osp.join(output_dir, output_filename)
        # 图片存在则保存标签
        generate_voc_xml_file(output_file, img_file, annotations)
    # save classes.txt
    # classes_file = osp.join(osp.dirname(output_dir), 'classes.txt')
    # save_classes(classes_file, class_names)
    # print msg
    LOGGER.info(f"dataset total class_names: {class_names}")
    # LOGGER.info(f"classes.txt saved in: {classes_file}")
    LOGGER.info(f"pascal voc type label saved in: {output_dir}")


if __name__ == "__main__":
    opt = parse_opt()
    project_root = opt.projection_root
    output_dir = opt.output_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if output_dir is None:
        output_dir = osp.join(osp.dirname(project_root), f'result_{timestamp}')
    output_image = osp.join(output_dir, 'JPEGImages')
    output_anno = osp.join(output_dir, 'Annotations')
    os.makedirs(output_image, exist_ok=True)
    os.makedirs(output_anno, exist_ok=True)
    project_names = os.listdir(project_root)
    print(project_names, '\n')
    # parser projection root
    for project_name in project_names:
        project_dir = osp.join(project_root, project_name)
        csv_file = osp.join(project_dir, 'key_object_final_results.csv')
        # copy image
        # camera 1
        copy_img(osp.join(project_dir, 'Camera1'), output_image)
        # camera 2
        copy_img(osp.join(project_dir, 'Camera2'), output_image)
        # csv to pascal voc xml
        run(csv_file=csv_file,
            img_dir=output_image,
            output_dir=output_anno)
        print(' ')
    LOGGER.info(f'all result saved in :{output_dir}')
