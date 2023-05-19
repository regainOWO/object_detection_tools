"""
本脚本的主要内容
根据label文件夹下的文件名称，进行随机采样，生成train val test三个列表。将其中的
内容分别存放到ImageSets/Main文件夹下train.txt、val.txt、test.txt和trainval.txt文件中。
格式类似与数据集PASCAL VOC的格式。
PASCAL VOC的格式的参考链接: https://blog.csdn.net/qq_37541097/article/details/115787033

trainval-percent: 训练集和验证集占总体的百分比
train-percent: 训练集占 训练集加验证集 总体的百分比
"""

import os
from pathlib import Path
import random
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-dir', type=str, required=True, help='input label label path')
    parser.add_argument('--save-dirname', type=str, default='Main', help='Sample data filename save path')
    parser.add_argument('--trainval-percent', type=float, default=0.9, help='dataset train and val label file nums percent in total')
    parser.add_argument('--train-percent', type=float, default=0.9, help='dataset train label file nums percent in train and val')
    opt = parser.parse_args()
    return opt


def split_train_val(label_dir, save_dirname, trainval_percent=0.9, train_percent=0.9):
    label_filenames = os.listdir(label_dir)
    save_dir = Path(label_dir).parent.joinpath('ImageSets', save_dirname).as_posix()
    os.makedirs(save_dir, exist_ok=True)

    num = len(label_filenames)
    index = range(num)
    trainval_num = int(num * trainval_percent)          # 训练和验证的数量
    train_num = int(trainval_num * train_percent)       # 训练的数量
    trainval = random.sample(index, trainval_num)       # 采样训练和验证集
    train = random.sample(trainval, train_num)          # 采样训练集

    ftrainval = open(save_dir + '/trainval.txt', 'w')
    ftest = open(save_dir + '/test.txt', 'w')
    ftrain = open(save_dir + '/train.txt', 'w')
    fval = open(save_dir + '/val.txt', 'w')

    for i in index:
        name = label_filenames[i][:-4]+'\n'     # label文件名称
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

    print(f"split dataset: {opt.label_dir} Success!!!, the voc imagesets file is in {save_dir}")


if __name__ == "__main__":
    opt = parse_opt()
    split_train_val(label_dir=opt.label_dir,
                    save_dirname=opt.save_dirname,
                    trainval_percent=opt.trainval_percent,
                    train_percent=opt.train_percent)