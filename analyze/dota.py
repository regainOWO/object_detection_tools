"""
分析dota格式数据的信息
"""
import argparse
import os
import os.path as osp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='D:/BaiduNetdiskDownload/SODA-A/SODA-A', help='input dataset dir')
    parser.add_argument('--label-dirname', type=str, default='labelTxt', help='label directory name')
    parser.add_argument('--output-dir', type=str, default=None, help='the output dir')
    opt = parser.parse_args()
    return opt


def plot_object_count(set_object_count: dict, output_dir):
    for k, v in set_object_count.items():
        o_file = osp.join(output_dir, f"{k}_object_count.jpg")
        names = list(v.keys())
        values = list(v.values())
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 13
        # 设置图大小
        plt.figure(figsize=(12, 4.8))
        # 绘制柱状图
        rect = plt.barh(names, values)
        # 条形图上写上数字
        for rec in rect:
            x = rec.get_y()
            height = rec.get_width()
            plt.text(1.02 * height, x, str(height))
        # 绘制标签
        plt.title(f"{k}标签类别数据")
        # 设置轴标签
        plt.xlabel("类别名称")
        plt.ylabel("数量")
        plt.savefig(o_file)
        plt.close()


def parse_dota(txt_file):
    with open(txt_file, mode='r', encoding='utf-8') as f:
        labels = [str(x).strip().split(' ') for x in f.readlines()]
    result = []
    names_count = {}
    for label in labels:
        class_name = label[-2]
        if class_name not in names_count:
            names_count[class_name] = 0
        difficult = int(label[-1])
        data = {
            "poly": list(map(float, label[:-2])),
            "name": class_name,
            "difficult": difficult
        }
        names_count[class_name] += 1
        result.append(data)
    return result, names_count


def parse_label(label_file):
    _, object_count = parse_dota(label_file)
    return object_count


def run(dataset_dir, label_dirname,  output_dir):
    assert osp.isdir(dataset_dir), f'路径: {dataset_dir} 文件夹不存在'
    if output_dir is None:
        output_dir = osp.join(dataset_dir, 'plot')
    os.makedirs(output_dir, exist_ok=True)

    set_object_count = {}
    for dir_name in ['train', 'val', 'test']:
        set_dir = osp.join(dataset_dir, dir_name)
        labels_dir = osp.join(set_dir, label_dirname)
        if not osp.isdir(labels_dir):
            continue
        total_object_count = {}
        for label_basename in tqdm(os.listdir(labels_dir)):
            label_file = osp.join(labels_dir, label_basename)
            # get label object count
            object_count = parse_label(label_file)
            # update total_object_count
            for name, count in object_count.items():
                if name not in total_object_count:
                    total_object_count[name] = count
                else:
                    total_object_count[name] += count
        total_object_count = dict(sorted(total_object_count.items(), key=lambda x: x[0], reverse=False))
        set_object_count[dir_name] = total_object_count
    plot_object_count(set_object_count, output_dir)



if __name__ == "__main__":
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        label_dirname=opt.label_dirname,
        output_dir=opt.output_dir)
