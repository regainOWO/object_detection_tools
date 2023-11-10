"""
此脚本的主要内容:
1. 将SODA-A的数据集格式转换成dota格式，并将图片赋值到指定的文件夹目录
2. 会过滤掉SODA-A中类别名称为ignore的标签
数据集目录
.
├── images
│   ├── ...
│   └── xxx.jpg
├── test
│   ├── ...
│   └── **.json
├── train
│   ├── ...
│   └── **.json
└── val
    ├── ...
    └── **.json

使用方法，需要输入三个参数：
--dataset-dir: SODA-A数据集的根路径
--output-dir: dota格式输出的路径，默认路径为数据集根路径的同级 SODA-A(dota)目录
"""
import argparse
import glob
import os
import os.path as osp
from pathlib import Path
import shutil

from tqdm import tqdm


try:
    import orjson as json
except ModuleNotFoundError:
    print("install orjson package makes read json file more quickly! ---->  \033[91mpip install orjson\033[0m")
    import json


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--output-dir', type=str, default=None, help='the output dir')
    opt = parser.parse_args()
    return opt


def json2dota(json_file, dst_txt_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    image_info = data['images']
    width, height = image_info['width'], image_info['height']
    
    category = {}
    for x in data['categories']:
        id = x['id']
        name = x['name']
        category[id] = name
    
    content_lines = []

    for annotation in data['annotations']:
        poly = annotation['poly']
        if len(poly) != 8:
            continue
        category_id = annotation['category_id']
        name = category[category_id]

        # 过滤掉ignore类型的标签
        if name == 'ignore':
            continue

        one_line = ' '.join([str(round(x, 1)) for x in poly]) + ' ' + name + ' ' + '0\n'
        content_lines.append(one_line)
    with open(dst_txt_file, 'w', encoding='utf-8') as f:
        f.writelines(content_lines)

def run(dataset_dir, output_dir):
    image_dir = osp.join(dataset_dir, 'images')
    assert osp.isdir(image_dir) and len(os.listdir(image_dir)), f'图片路径: {image_dir} 不是文件夹或文件夹内容为空'
    if output_dir is None:
        output_dir = osp.join(osp.dirname(dataset_dir), 'SODA-A(dota)')
    os.makedirs(output_dir, exist_ok=True)
    for dirname in ['train', 'val', 'test']:
        label_dir = osp.join(dataset_dir, dirname)
        if not osp.isdir(label_dir):
            continue
        # 
        label_files = glob.glob(label_dir + '/*.json')
        image_files = [Path(image_dir).joinpath(osp.basename(x)).with_suffix('.jpg') for x in label_files]
        
        dst_dir = osp.join(output_dir, dirname)
        dst_label_dir = osp.join(dst_dir, 'labelTxt')
        dst_image_dir = osp.join(dst_dir, 'images')
        os.makedirs(dst_label_dir, exist_ok=True)
        os.makedirs(dst_image_dir, exist_ok=True)
        # 转换标签
        for image_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files)):
            if not osp.isfile(image_file):  # 图片文件
                continue
            basename = osp.basename(image_file)
            filename = osp.splitext(basename)[0]
            dst_label_file = osp.join(dst_label_dir, filename + '.txt')
            dst_image_file = osp.join(dst_image_dir, basename)
            # 转换标签
            json2dota(label_file, dst_label_file)
            # 复制图片
            shutil.copy(image_file, dst_image_file)


if __name__ == "__main__":
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        output_dir=opt.output_dir)
    