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
--output-dir: yolo格式输出的路径，默认路径为labelme标签文件夹的同级 annotations目录

"""

import argparse
import glob
import json
import logging
import shutil
import time
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
    parser.add_argument('--project-root', type=str, required=True,
                        help='根路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出路径')
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
        bboxs2 = eval(row['bboxs2'])
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

    pbar = tqdm(data_info.items(), total=len(data_info), desc='save json: ')
    # 标签文件转化
    for img_file, annotations in pbar:
        output_filename = osp.splitext(osp.basename(img_file))[0] + '.json'
        output_file = osp.join(output_dir, output_filename)
        # 图片存在则保存标签
        generate_json_file(output_file, img_file, annotations)
    # save classes.txt
    # classes_file = osp.join(osp.dirname(output_dir), 'classes.txt')
    # save_classes(classes_file, class_names)
    # print msg
    LOGGER.info(f"dataset total class_names: {class_names}")
    # LOGGER.info(f"classes.txt saved in: {classes_file}")
    LOGGER.info(f"pascal voc type label saved in: {output_dir}")


if __name__ == "__main__":
    opt = parse_opt()
    project_root = opt.project_root
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
    # parser project root
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
    LOGGER.info(f'all result saved in :{output_dir}')
