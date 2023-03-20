"""
此脚本的主要内容:
将coco的数据集格式转换成yolo格式，目前只支持转"目标检测的格式"

使用方法，需要输入四个参数：
--dataset-dir: 数据集的根路径。
--trainimg-dirname: 数据集根路径下，存放训练集图片的文件夹名称，默认值为train2017
--valimg-dirname: 数据集根路径下，存放验证集图片的文件夹名称，默认值为val2017
--trainjson-filename: 数据集根路径下annotations文件夹下，训练集标签json文件的名称，默认值为instances_train2017.json
--valjson-filename: 数据集根路径下annotations文件夹下，训练集标签json文件的名称，默认值为instances_val2017.json
--output-dir: 生成新数据集的路径，默认值为None，指定了值，会对图片文件进行复制，不推荐。

转化的结果分成两种
1. --output-dir 未指定值 （前提：训练集图片和验证集图片文件名称不能同名）
获将对应的图片移动到对应文件夹中，例如train2017中的图片会移动到train/images文件夹中，且在数据集根路径下会创建train.txt和val.txt，里面存放了对应数据集的图片的绝对路径
.
├── annotations
├── train
│   ├── images
│   ├── labels
│   └── classes.txt
├── val
│   ├── images
│   ├── labels
│   └── classes.txt
├── train2017
├── val2017
├── xxx.yaml
├── train.txt
└── val.txt

2. --output-dir 指定了值
.
├── train
│   ├── images
│   ├── labels
│   └── classes.txt
├── val
│   ├── images
│   ├── labels
│   └── classes.txt
├── xxx.yaml
├── train.txt
└── val.txt

"""
import argparse
import os
import os.path as osp
import json
import shutil
from tqdm import tqdm
from pathlib import Path

import yaml


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--trainimg-dirname', type=str, default="train2017", help='train image directory name under the dataset directory, default is train2017')
    parser.add_argument('--valimg-dirname', type=str, default="val2017", help='train image directory name under the dataset directory, default is val2017')
    parser.add_argument('--trainjson-filename', type=str, default="instances_train2017.json", help='train label .json filename under the annotations directory, default is instances_train2017.json')
    parser.add_argument('--valjson-filename', type=str, default="instances_val2017.json", help='val label .json filename under the annotations directory, default is instances_val2017.json')
    parser.add_argument('--output-dir', type=str, default=None, help='new dataset dir, it will copy image, not recommend !!')
    opt = parser.parse_args()
    return opt


# ltwh2xywh 并归一化
def ltwh2xywh_normalize(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def coco2yolo(json_file: str, labels_dir: str):
    data = json.load(open(json_file, 'r'))

    id_map = {}
    names = {}

    # 解析目标类别，也就是 categories 字段，并将类别写入文件 classes.txt 中，存放在label_dir的同级目录中
    data_dir = Path(labels_dir).parent.as_posix()
    with open(osp.join(data_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
            names[i] = category['name']
    print(f"generate classes.txt under the {data_dir} Success!!")

    img_filenames = []
    for img in tqdm(data['images'],total=len(data['images']), desc="convert data...."):

        # 解析 images 字段，分别取出图片文件名、图片的宽和高、图片id
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]

        # label文件名，与对应图片名只有后缀名不一样
        label_filename = osp.splitext(filename)[0] + ".txt"
        label_file = osp.join(labels_dir, label_filename)
        content_lines = []
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = ltwh2xywh_normalize((img_width, img_height), ann["bbox"])

                # 写入txt，共5个字段
                content_lines.append("%s %s %s %s %s\n" % (
                    id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        # 将图片的标签写入到文件中
        with open(label_file, 'w', encoding='utf-8') as f:
            f.writelines(content_lines)
        img_filenames.append(filename)
    
    return img_filenames, names


def save_yolo_data_config(dataset_dir, names):
    # Save dataset.yaml
    d = {'path': osp.abspath(dataset_dir),
         'train': "images/train",
         'val': "images/val",
         'test': None,
         'nc': len(names.keys()),       # yolov5 later
         'names': names}  # dictionary

    with open(osp.join(dataset_dir, Path(dataset_dir).with_suffix('.yaml').name), 'w', encoding='utf-8') as f:
        yaml.dump(d, f, sort_keys=False)
    print(f"generate yolo yaml file under the {dataset_dir} Success!!")


def save_txt(save_path, files):
    files = [x + '\n' for x in files]
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(files)
    print(f"generate {save_path} Success!!")


def cp_images(img_filenames, o_dir, d_dir, tag='train'):
    dst_img_files = []
    for img_filename in tqdm(img_filenames, total=len(img_filenames), desc=f"copy {tag} image...."):
        o_img_file = osp.join(o_dir, img_filename)
        d_img_file = osp.join(d_dir, img_filename)
        shutil.copy(o_img_file, d_img_file)
        dst_img_files.append(d_img_file)
    return dst_img_files


def mv_images(img_filenames, o_dir, d_dir, tag='train'):
    dst_img_files = []
    for img_filename in tqdm(img_filenames, total=len(img_filenames), desc=f"move {tag} image...."):
        o_img_file = osp.join(o_dir, img_filename)
        d_img_file = osp.join(d_dir, img_filename)
        shutil.move(o_img_file, d_img_file)
        dst_img_files.append(d_img_file)
    return dst_img_files


def new_dataset(output_dir, o_train_img_dir, o_val_img_dir, train_json_file, val_json_file):
    # 删除old，创建输出路径的文件夹
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    # 转成绝对路径
    output_dir = osp.abspath(output_dir)

    # train
    train_dir = osp.join(output_dir, 'train')
    d_train_img_dir = osp.join(train_dir, 'images')
    d_train_label_dir = osp.join(train_dir, 'labels')
    os.makedirs(d_train_img_dir)
    os.makedirs(d_train_label_dir)
    # convert train label
    print(f"start to convert train annotation file {train_json_file}.....")
    train_img_filenames, names = coco2yolo(train_json_file, d_train_label_dir)
    print(f"convert {train_json_file} Success!! ")
    # copy train image
    train_dst_img_files = cp_images(train_img_filenames, o_train_img_dir, d_train_img_dir, tag='train')

    # val
    val_dir = osp.join(output_dir, 'val')
    d_val_img_dir = osp.join(val_dir, 'images')
    d_val_label_dir = osp.join(val_dir, 'labels')
    os.makedirs(d_val_img_dir)
    os.makedirs(d_val_label_dir)
    # convert val label
    print(f"start to convert val annotation file {val_json_file}")
    val_img_filenames, names = coco2yolo(val_json_file, d_val_label_dir)
    print(f"convert {val_json_file} Success!! ")
    # copy val image
    val_dst_img_files = cp_images(val_img_filenames, o_val_img_dir, d_val_img_dir, tag='val')

    # Save dataset.yaml
    save_yolo_data_config(output_dir, names)

    # save train.txt, val.txt for image abs path
    # train.txt
    save_txt(osp.join(output_dir, 'train.txt'), train_dst_img_files) 
    # val.txt
    save_txt(osp.join(output_dir, 'val.txt'), val_dst_img_files)


def just_convertlabel(dataset_dir, o_train_img_dir, o_val_img_dir, train_json_file, val_json_file):
    # 转成绝对路径
    dataset_dir = osp.abspath(dataset_dir)
    # train
    train_dir = osp.join(dataset_dir, 'train')
    d_train_img_dir = osp.join(train_dir, 'images')
    d_train_label_dir = osp.join(train_dir, "labels")
    os.makedirs(d_train_img_dir, exist_ok=True)
    os.makedirs(d_train_label_dir, exist_ok=True)
    # convert train label
    print(f"start to convert train annotation file {train_json_file}.....")
    train_img_filenames, names = coco2yolo(train_json_file, d_train_label_dir)
    print(f"convert {train_json_file} Success!! ")
    # move train images
    train_dst_img_files = mv_images(train_img_filenames, o_train_img_dir, d_train_img_dir, tag='train')

    # val
    val_dir = osp.join(dataset_dir, 'val')
    d_val_img_dir = osp.join(val_dir, 'images')
    d_val_label_dir = osp.join(val_dir, "labels")
    os.makedirs(d_val_img_dir, exist_ok=True)
    os.makedirs(d_val_label_dir, exist_ok=True)
    # convert train label
    print(f"start to convert val annotation file {val_json_file}")
    val_img_filenames, names = coco2yolo(val_json_file, d_val_label_dir)
    print(f"convert {val_json_file} Success!! ")
    # move val images
    val_dst_img_files = mv_images(val_img_filenames, o_val_img_dir, d_val_img_dir, tag='val')

    # Save dataset.yaml
    save_yolo_data_config(dataset_dir, names)

    # save train.txt, val.txt for image abs path
    # train.txt
    save_txt(osp.join(dataset_dir, 'train.txt'), train_dst_img_files) 
    # val.txt
    save_txt(osp.join(dataset_dir, 'val.txt'), val_dst_img_files)


def run(dataset_dir, train_img_dirname, val_img_dirname, train_json_filename, val_json_filename, output_dir=None):
    train_json_file = osp.join(dataset_dir, "annotations", train_json_filename)
    val_json_file = osp.join(dataset_dir, "annotations", val_json_filename)
    train_img_dir = osp.join(dataset_dir, train_img_dirname)
    val_img_dir = osp.join(dataset_dir, val_img_dirname)
    assert osp.exists(train_json_file), f"{train_json_file} not exists, please make sure your input param trainjson_filename is correct"
    assert osp.exists(val_json_file), f"{val_json_file} not exists, please make sure your input param valjson_filename is correct"
    assert osp.exists(train_img_dir), f"{train_img_dir} not exists, please make sure your trainimg_dirname is correct"
    assert osp.exists(val_img_dir), f"{val_img_dir} not exists, please make sure your valimg_dirname is correct"
    if output_dir is None:
        just_convertlabel(dataset_dir, train_img_dir, val_img_dir, train_json_file, val_json_file)
    else:
        new_dataset(output_dir, train_img_dir, val_img_dir, train_json_file, val_json_file)


if __name__ == '__main__':
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        train_img_dirname=opt.trainimg_dirname,
        val_img_dirname=opt.valimg_dirname,
        train_json_filename=opt.trainjson_filename,
        val_json_filename=opt.valjson_filename,
        output_dir=opt.output_dir)
