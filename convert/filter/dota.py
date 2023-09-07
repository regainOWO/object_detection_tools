"""
此脚本的主要内容:
将dota的数据集格式，根据类别名称进行过滤，只保留包含类的标签

使用方法，需要输入四个参数：
--dataset-dir: 数据集的根路径。同时要确保classes.txt在该目录下，输出数据集label的索引值见按照该文件中的内容来确定。
--image-dirname: 图片的文件夹名称，默认值为images（图片文件夹需要放在数据集的根文件夹中）
--label-dirname: dota标签的文件夹名称，默认值为labelTxt（标签文件夹需要放在数据集的根文件夹中）
--difficult-thres: dota标签中每个标签都还有一个difficult值，用于表示识别的难易度，值越大，越难识别。
该参数的默认值为2，difficult的值若小于该值，则将标签保留下来，反之亦然
--output-dir: 存放的路径
--classes: 类别名称
--save_empty: 是否保存空标签的数据
"""

import argparse
import glob
import os
import os.path as osp
import shutil
from pathlib import Path

from PIL.Image import init, EXTENSION
from tqdm import tqdm

if not len(EXTENSION):
    init()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--image-dirname', type=str, default='images',
                        help='the image directory name under the dataset directory')
    parser.add_argument('--label-dirname', type=str, default='labelTxt',
                        help='the dota label directory name under the dataset directory')
    parser.add_argument('--difficult-thres', type=int, default=1,
                        help='skip the lable by difficult less than or equal to difficult-thres')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('--classes', type=str, nargs='+', default=None, help='The reserved class name to filter dataset, default nothing to do')
    parser.add_argument('--save_empty', action='store_true', help='whether save the empty data')
    opt = parser.parse_args()
    return opt


def get_path_basename(path):
    """获取文件路径的文件名称，不包含拓展名"""
    p = Path(path)
    return p.name.split(p.suffix)[0]


def dota_filter(in_file, out_file, classes, difficult_thres, save_empty=False):
    with open(in_file, mode='r', encoding='utf-8') as f:
        labels = [str(x).strip().split(' ') for x in f.readlines()]
    content_lines = []
    for i, label in enumerate(labels):
        class_name = label[-2]
        difficult = int(label[-1])
        if class_name not in classes:
            continue
        elif difficult > difficult_thres:
            print(f"ignore {in_file} line {i}, because difficult is greater than {difficult_thres}")
            continue
        one_line = ' '.join(label) + '\n'
        content_lines.append(one_line)

    if len(content_lines) or save_empty:
        # 将内容写入文件
        with open(out_file, mode='w', encoding='utf-8') as f:
            f.writelines(content_lines)
        return True
    return False


def run(dataset_dir, image_dirname, label_dirname, difficult_thres, classes: list, output_dir, save_empty=False):
    assert classes, 'classes param is None, the --classes param must be set'
    assert difficult_thres >= 0, "the param difficult_thres must be greate than or equal to 0"
    label_dir = osp.join(dataset_dir, label_dirname)
    image_dir = osp.join(dataset_dir, image_dirname)
    label_files = [osp.join(label_dir, x) for x in os.listdir(label_dir)]
    if output_dir is None:
        output_dir = osp.join(dataset_dir, 'filter')
        print(f"output_dir param is None, auto set output_dir is: {output_dir}")
    o_image_dir = osp.join(output_dir, 'images')
    o_label_dir = osp.join(output_dir, 'labelTxt')
    os.makedirs(o_image_dir, exist_ok=True)
    os.makedirs(o_label_dir, exist_ok=True)

    for label_file in tqdm(label_files, total=len(label_files)):
        o_label_file = osp.join(o_label_dir, Path(label_file).name)
        # 在图片文件夹中寻找相同文件名称的图片文件
        file_basename = get_path_basename(label_file)
        try:
            image_file = glob.glob(osp.join(image_dir, file_basename + '.*'))[0]
        except Exception as e:
            print(f"please make sure the folder: {image_dir} have correct image named like {file_basename}")
            continue
        if osp.splitext(image_file)[-1] not in EXTENSION:
            print(f'find file: {image_file}, but is not an image format')
            continue
        label_saved = dota_filter(label_file, o_label_file, classes, difficult_thres)
        o_image_file = osp.join(o_image_dir, osp.basename(image_file))
        if label_saved:
            shutil.copy(image_file, o_image_file)

    # print msg
    print(f"filter dataset: {Path(dataset_dir).name} dota label Success!!!")
    print(f"the result is in folder {output_dir}")


if __name__ == '__main__':
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        image_dirname=opt.image_dirname,
        label_dirname=opt.label_dirname,
        difficult_thres=opt.difficult_thres,
        classes=opt.classes,
        output_dir=opt.output_dir,
        save_empty=opt.save_empty)