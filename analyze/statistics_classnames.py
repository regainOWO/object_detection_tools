"""
此脚本的主要内容:
统计目标检测数据集中的标签内容，目前支持coco、pascalvoc、yolo、dota格式

输入参数:
--label-dir: 标签文件的文件夹路径
--label-type: 标签的格式类型，目前支持coco、pascalvoc、yolo、dota格式
"""
import argparse
import json
import os
import os.path as osp
from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm

DETECTION_LABEL_TYPE = ['coco', 'pascal', 'yolo', 'dota']


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-dir', type=str, required=True, help='标签文件夹的路径')
    parser.add_argument('--label-type', type=str, default='coco', help='标签的格式类型，目前支持coco、pascalvoc、yolo、dota格式')
    opt = parser.parse_args()
    return opt


def statistics_coco(label_dir):
    label_files = [osp.join(label_dir, x) for x in os.listdir(label_dir) if osp.splitext(x)[-1] == '.json']
    assert len(label_files), f"{label_dir} 目录下没有coco的 .json 标签文件"
    pbar = tqdm(label_files, total=len(label_files))
    results = []
    ids = []
    for label_file in pbar:
        pbar.desc = f"正在解析{label_file}"
        with open(label_file, mode='r', encoding='utf-8') as f:
            data = json.load(f)
        # category
        id_map = {}
        names = {}
        supercategorys = {}
        categories = data.get('categories', None)
        if categories is None or len(categories) <= 0:
            print(f"{label_file} 文件没有categories属性 或categories为空, 请确保文件格式是coco的标准格式")
            continue
        for i, category in enumerate(categories):
            id_map[category['id']] = i
            names[i] = category['name']
            supercategorys[i] = category['supercategory']
        # annotation
        annotations = data.get('annotations', None)
        if annotations is None or len(annotations) <= 0:
            print(f"{label_file} 文件没有annotations属性 或annotations属性为空, 请确保文件格式是coco的标准格式")
            continue
        for ann in annotations:
            id = id_map[ann["category_id"]]
            result = {
                "id": id,
                "name": names[id],
                "supercategory": supercategorys[id]
            }
            if result in results:
                continue
            ids.append(id)
            results.append(result)
    # sorted
    rst = {x['id']: x for x in results}
    rst = sorted(rst.items(), key=lambda i: int(i[0]))
    results = rst.values()
    return results


def statistics_pascal(label_dir):
    from xml.etree import ElementTree as ET
    label_files = [osp.join(label_dir, x) for x in os.listdir(label_dir) if osp.splitext(x)[-1] == '.xml']
    assert len(label_files), f"{label_dir} 目录下没有pascal的 .xml 标签文件"
    pbar = tqdm(label_files, total=len(label_files))
    results = []
    for label_file in pbar:
        pbar.desc = f"正在解析{label_file}"
        with open(label_file, mode='r', encoding='utf-8') as f:
            tree = ET.parse(label_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            name = obj.find('name').text
            result = {
                "id": None,
                "name": name,
                "supercategory": None
            }
            if result in results:
                continue
            results.append(result)
    return results


def statistics_yolo(label_dir):
    label_files = [osp.join(label_dir, x) for x in os.listdir(label_dir) if osp.splitext(x)[-1] == '.txt']
    assert len(label_files), f"{label_dir} 目录下没有yolo的 .txt 标签文件"
    pbar = tqdm(label_files, total=len(label_files))
    results = []
    for label_file in pbar:
        pbar.desc = f"正在解析{label_file}"
        with open(label_file, mode='r', encoding='utf-8') as f:
            lines = [x.strip() for x in f.readlines()]
        for line in lines:
            cls_idx = line.split(' ')[0]
            result = {
                "id": int(cls_idx),
                "name": None,
                "supercategory": None
            }
            if result in results:
                continue
            results.append(result)
    return results


def statistics_dota(label_dir):
    label_files = [osp.join(label_dir, x) for x in os.listdir(label_dir) if osp.splitext(x)[-1] == '.txt']
    assert len(label_files), f"{label_dir} 目录下没有dota的 .txt 标签文件"
    pbar = tqdm(label_files, total=len(label_files))
    results = []
    for label_file in pbar:
        pbar.desc = f"正在解析{label_file}"
        with open(label_file, mode='r', encoding='utf-8') as f:
            lines = [x.strip() for x in f.readlines()]
        for line in lines:
            name = line.split(' ')[-1]
            result = {
                "id": None,
                "name": name,
                "supercategory": name
            }
            if result in results:
                continue
            results.append(result)
    return results


def run(label_dir, label_type):
    assert osp.isdir(label_dir), f"{label_dir} 不存在或不是一个文件夹"
    assert label_type in DETECTION_LABEL_TYPE, f"暂不支持 {label_type}格式的数据统计"

    if label_type == 'coco':
        categories = statistics_coco(label_dir)
    elif label_type == 'pascal':
        categories = statistics_pascal(label_dir)
    elif label_type == 'yolo':
        categories = statistics_yolo(label_dir)
    elif label_type == 'dota':
        categories = statistics_dota(label_dir)
    else:
        assert 0, f"暂不支持 {label_type}格式的数据统计"
    save_json = {"categories": categories}

    with open(Path(label_dir).parent.joinpath('classes.json'), mode='w', encoding='utf-8') as f:
        json.dump(save_json, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    opt = parse_opt()
    run(label_dir=opt.label_dir,
        label_type=opt.label_type)
