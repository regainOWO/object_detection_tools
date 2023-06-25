"""
来源与全国大数据与计算机智能挑战赛-基于亚米级影像的精细化目标检测
将csv文件中的dota标签内容转成txt文件格式，参数如下:

--csv-path: csv标签文件的路径
--output-path: txt标签文件的输出路径
--separator: 每列数据间的分隔符，默认值为空格
"""

import os
import os.path as osp
import argparse
from pathlib import Path

import pandas
from tqdm import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str, required=True, help='csv标签文件夹的路径')
    parser.add_argument('--output-dir', type=str, default=None, help='标签输出的路径')
    parser.add_argument('--separator', type=str, default=' ', help='每列数据间的分隔符，默认值为空格')
    opt = parser.parse_args()
    return opt


def run(csv_file, output_dir, separator=' '):
    df = pandas.read_csv(csv_file)
    if output_dir is None:
        output_dir = "labels"
    else:
        output_dir = osp.join(output_dir, 'labels')
    os.makedirs(output_dir, exist_ok=True)

    class_names = []

    colunm_names = df.columns.tolist()
    total_data = {}
    process_csv_pbar = tqdm(df.iterrows(), desc=f"preprocess with the file: {csv_file}")
    for index, row in process_csv_pbar:
        difficulty = 0
        image_id = row['ImageID']
        label_name = row['LabelName']
        if label_name not in class_names:
            class_names.append(label_name)
        x1, y1 = row['X1'], row['Y1']
        x2, y2 = row['X2'], row['Y2']
        x3, y3 = row['X3'], row['Y3']
        x4, y4 = row['X4'], row['Y4']
        one_line = str(x1) + separator + str(y1) + separator + str(x2) + separator + str(y2) + separator + \
                   str(x3) + separator + str(y3) + separator + str(x4) + separator + str(y4) + separator + \
                   str(label_name) + separator + str(difficulty) + '\n'
        if image_id not in total_data:
            total_data[image_id] = []
        total_data[image_id].append(one_line)

    classes_txt_file = Path(output_dir).parent.joinpath('classes.txt')
    class_names.sort()  # 先排序
    with open(classes_txt_file, mode='w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in class_names])
    print(f"标签文件的所有类别已保存至: {osp.abspath(classes_txt_file)}")

    process_label_pbar = tqdm(total_data.items(), total=len(total_data.items()))
    for image_id, datas in process_label_pbar:
        label_filename = osp.splitext(image_id)[0] + '.txt'
        label_path = osp.join(output_dir, label_filename)
        process_label_pbar.desc = f"save the label file: {osp.abspath(label_path)}"
        with open(label_path, mode='w', encoding='utf-8') as f:
            f.writelines(datas)
    print(f"标签文件已保存至: {osp.abspath(output_dir)}")


if __name__ == '__main__':
    opt = parse_opt()
    run(csv_file=opt.csv_path,
        output_dir=opt.output_dir,
        separator=opt.separator)
