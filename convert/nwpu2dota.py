"""
此脚本的主要内容
* 将NWPU VHR-10的数据格式转换成dota格式

数据集目录
.
├── ground truth
│   ├── ...
│   └── **.txt

使用方法，需要输入两个参数：
--label-dir: 标签文件存放目录
--output-dir: 转化后dota格式标签存放的目录，默认值为label-dir参数的统计目录labelTxt
"""
import argparse
import glob
import os
import os.path as osp
from pathlib import Path

from tqdm import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-dir', type=str, required=True,
                        help='annotation directory name under the dataset dir')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output directory path, default is labelTxt directory with the label dir')
    opt = parser.parse_args()
    return opt


def replace_bracket(value: str):
    value = value.replace('(', '')
    value = value.replace(')', '')
    value = value.strip()
    return value


def nwpu2dota(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = [x.strip() for x in f.readlines()]

    image_annotations = []
    class_names = []
    for line in lines:
        if len(line.strip()) == 0:
            continue
        data = line.split(',')
        class_name = replace_bracket(data[4])
        xyxy = map(replace_bracket, data[:4])
        xmin, ymin, xmax, ymax = map(float, xyxy)
        annotation = {
            'poly': [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin],
            'name': class_name,
            'difficult': 0
        }
        image_annotations.append(annotation)
        if class_name not in class_names:
            class_names.append(class_name)
    return image_annotations, class_names


def nwpu2dota_txt(txt_file, out_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = [x.strip() for x in f.readlines()]

    contents = []
    for line in lines:
        if len(line.strip()) == 0:
            continue
        data = line.split(',')
        class_name = replace_bracket(data[4])
        xyxy = map(replace_bracket, data[:4])
        xmin, ymin, xmax, ymax = map(float, xyxy)
        poly = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
        one_line = ' '.join([str(x) for x in poly]) + ' ' + class_name + ' ' + '0\n'
        contents.append(one_line)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.writelines(contents)


def run(label_dir: str, output_dir: str = None):
    assert osp.isdir(label_dir), f"{label_dir} 目录不存在"
    label_files = glob.glob(label_dir + '/*.txt')
    if output_dir is None:
        output_dir = Path(label_dir).parent.joinpath('labelTxt').as_posix()
    os.makedirs(output_dir, exist_ok=True)

    for label_file in tqdm(label_files):
        label_filename = osp.basename(label_file)
        out_file = osp.join(output_dir, label_filename)
        nwpu2dota_txt(label_file, out_file)


if __name__ == '__main__':
    opt = parse_opt()
    run(label_dir=opt.label_dir,
        output_dir=opt.output_dir)
