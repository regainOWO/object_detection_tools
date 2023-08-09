"""
此脚本的主要内容
* 将信息工程大学的数据格式转换成dota格式

数据集目录
.
├── xxx.tif
├── xxx.json
├── ...
└── ***.tif/json

使用方法，需要输入两个参数：
--label-dir: 标签文件存放目录
--output-dir: 转化后dota格式标签存放的目录，默认值为label-dir参数的同级目录labelTxt
"""
import argparse
import glob
import json
import os
import os.path as osp

from tqdm import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-dir', type=str, required=True,
                        help='annotation directory name under the dataset dir')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output directory path, default is labelTxt directory with the label dir')
    opt = parser.parse_args()
    return opt


def coordinates_geographic2pixel(geo_transform, geo_x, geo_y):
    """
    将投影或地理坐标转为影像上的像素坐标
    Args:
        geo_transform (list): 坐标系转换的矩阵
        geo_x (float): 投影或地理坐标x
        geo_y (float): 投影或地理坐标y
    Returns:
        px (int): 像素坐标x
        py (int): 像素坐标y
    """
    geo_tl_x = geo_transform[0]              # 0
    geo_tl_y = geo_transform[3]              # 1024
    pixel_width = geo_transform[1]           # 1
    pixel_height = geo_transform[5]          # -1
    rotate1 = geo_transform[2]               # 0
    rotate2 = geo_transform[4]               # 0

    temp = pixel_width * pixel_height - rotate1 * rotate2
    px = int((pixel_height * (geo_x - geo_tl_x) - rotate1 * (geo_y - geo_tl_y)) / temp + 0.5)
    py = int((pixel_width * (geo_y - geo_tl_y) - rotate2 * (geo_x - geo_tl_x)) / temp + 0.5)
    return px, py


def ie_university2dota(json_file):
    with open(json_file, mode='r', encoding='utf-8') as f:
        data = json.loads(f.read())

    geo_trans = data['geoTrans']
    width, height = data['imageWidth'], data['imageHeight']

    image_annotations = []
    class_names = []
    for shape in data['shapes']:
        class_name = shape['label']
        probability = shape['probability']
        points = shape['points']
        difficult = 0 if probability > 8 else 1
        poly = []
        for point in points:
            px, py = coordinates_geographic2pixel(geo_trans, point[0], point[1])
            poly.append(px)
            poly.append(py)
        annotation = {
            'poly': poly,
            'name': class_name,
            'difficult': difficult,
        }
        image_annotations.append(annotation)
        if class_name not in class_names:
            class_names.append(class_name)
    return image_annotations, class_names


def ie_university2dota_txt(json_file, dst_file):
    with open(json_file, mode='r', encoding='utf-8') as f:
        data = json.loads(f.read())

    geo_trans = data['geoTrans']
    width, height = data['imageWidth'], data['imageHeight']

    content_lines = []
    for shape in data['shapes']:
        class_name = shape['label']
        probability = shape['probability']
        points = shape['points']
        difficult = 0 if probability > 8 else 1
        poly = []
        for point in points:
            px, py = coordinates_geographic2pixel(geo_trans, point[0], point[1])
            poly.append(px)
            poly.append(py)
        one_line = ' '.join([str(round(x, 1)) for x in poly]) + ' ' + str(class_name) + ' ' + str(difficult) + '\n'
        content_lines.append(one_line)
    with open(dst_file, mode='w', encoding='utf-8') as f:
        f.writelines(content_lines)


def run(label_dir, output_dir=None):
    assert osp.isdir(label_dir), f"目录 {label_dir}不存在"
    label_files = glob.glob(label_dir + '/*.json')
    assert len(label_files), f"目录 {label_dir}下没有标签.json文件"
    if output_dir is None:
        output_dir = osp.join(osp.dirname(osp.abspath(label_dir)), 'labelTxt')
    if not osp.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for label_file in tqdm(label_files):
        filename = osp.basename(label_file)
        filename = osp.splitext(filename)[0] + '.txt'
        dst_file = osp.join(output_dir, filename)
        ie_university2dota_txt(label_file, dst_file)


if __name__ == '__main__':
    opt = parse_opt()
    run(label_dir=opt.label_dir,
        output_dir=opt.output_dir)

