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
会将所有的标签存放到数据集根路径下的labels文件夹中，且在数据集根路径下会创建train.txt和val.txt，里面存放了对应数据集的图片的绝对路径
.
├── annotations
├── labels
│   └── xxx.txt
├── train2017
│   └── xxx.jpg
├── val2017
│   └── xxx.jpg
├── classes.txt
├── train.txt
└── val.txt

2. --output-dir 指定了值
.
├── train
│   ├── images
│   ├── labels
│   └── classes.txt
└── val
    ├── images
    ├── labels
    └── classes.txt

"""
import argparse
import os
import os.path as osp
import json
import shutil
from tqdm import tqdm
from pathlib import Path


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

    # 解析目标类别，也就是 categories 字段，并将类别写入文件 classes.txt 中，存放在label_dir的同级目录中
    data_dir = Path(labels_dir).parent.as_posix()
    with open(osp.join(data_dir, 'classes.txt'), 'w') as f:
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
    print(f"generate classes.txt under the {data_dir} Success!!")

    img_filenames = []
    for img in tqdm(data['images'],total=len(data['images'])):

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


def new_dataset(output_dir, o_train_img_dir, o_val_img_dir, train_json_file, val_json_file):
    # 转成绝对路径
    o_train_img_dir = osp.abspath(o_train_img_dir)
    o_val_img_dir = osp.abspath(o_val_img_dir)

    # train
    train_dir = osp.join(output_dir, 'train')
    train_images_dir = osp.join(train_dir, 'images')
    train_labels_dir = osp.join(train_dir, 'labels')
    os.makedirs(train_images_dir)
    os.makedirs(train_labels_dir)

    # convert train label
    print("start to convert train annotation file %s.....".format(train_json_file))
    train_img_filenames = coco2yolo(train_json_file, train_labels_dir)
    print("convert %s Success!! ".format(train_json_file))
    # copy train image
    for img_filename in tqdm(train_img_filenames, total=len(train_img_filenames), desc="copy train image...."):
        o_img_file = osp.join(o_train_img_dir, img_filename)
        d_img_file = osp.join(train_images_dir, img_filename)
        shutil.copy(o_img_file, d_img_file)

    # val
    val_dir = osp.join(output_dir, 'val')
    val_images_dir = osp.join(val_dir, 'images')
    val_labels_dir = osp.join(val_dir, 'labels')
    os.makedirs(val_images_dir)
    os.makedirs(val_labels_dir)
    # convert val label
    print("start to convert val annotation file %s".format(val_json_file))
    val_img_filenames = coco2yolo(val_json_file, val_labels_dir)
    print("convert %s Success!! ".format(val_json_file))

    # copy val image
    for img_filename in tqdm(val_img_filenames, total=len(val_img_filenames), desc="copy val image...."):
        o_img_file = osp.join(o_val_img_dir, img_filename)
        d_img_file = osp.join(val_images_dir, img_filename)
        shutil.copy(o_img_file, d_img_file)



def just_convertlabel(dataset_dir, train_img_dir, val_img_dir, train_json_file, val_json_file):
    labels_dir = osp.join(dataset_dir, "labels")
    # 删除old，创建输出路径的文件夹
    if osp.exists(labels_dir):
        shutil.rmtree(labels_dir)
    os.makedirs(labels_dir)
    # 转成绝对路径
    train_img_dir = osp.abspath(train_img_dir)
    val_img_dir = osp.abspath(val_img_dir)
    # train
    print("start to convert train annotation file %s.....".format(train_json_file))
    train_img_filenames = coco2yolo(train_json_file, labels_dir)
    print("convert %s Success!! ".format(train_json_file))
    # val
    print("start to convert val annotation file %s".format(val_json_file))
    val_img_filenames = coco2yolo(val_json_file, labels_dir)
    print("convert %s Success!! ".format(val_json_file))

    # image file path txt
    # train.txt
    train_img_files = [osp.join(train_img_dir, x + '\n') for x in train_img_filenames]
    with open(osp.join(dataset_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_img_files)
    print(f"generate train.txt under the {dataset_dir} Success!!")
    # val.txt
    val_img_files = [osp.join(val_img_dir, x + '\n') for x in val_img_filenames]
    with open(osp.join(dataset_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.writelines(val_img_files)
    print(f"generate val.txt under the {dataset_dir} Success!!")


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
        # 删除old，创建输出路径的文件夹
        if osp.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        new_dataset(output_dir, train_img_dir, val_img_dir, train_json_file, val_json_file)


if __name__ == '__main__':
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        train_img_dirname=opt.trainimg_dirname,
        val_img_dirname=opt.valimg_dirname,
        train_json_filename=opt.trainjson_filename,
        val_json_filename=opt.valjson_filename,
        output_dir=opt.output_dir)
