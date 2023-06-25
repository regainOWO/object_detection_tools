"""
本脚本的主要内容
根据label文件夹下的文件名称，进行随机采样，生成train val test三个txt文件，里面存放都是对图片的绝对路径。
格式类似与数据集PASCAL VOC的格式。
PASCAL VOC的格式的参考链接: https://blog.csdn.net/qq_37541097/article/details/115787033

image-dir: 图片文件夹的路径
trainval-percent: 训练集和验证集占总体的百分比
train-percent: 训练集占 训练集加验证集 总体的百分比
"""

import os
from pathlib import Path
import random
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True, help='image directory path')
    parser.add_argument('--trainval-percent', type=float, default=0.9, help='dataset train and val label file nums percent in total')
    parser.add_argument('--train-percent', type=float, default=0.9, help='dataset train label file nums percent in train and val')
    opt = parser.parse_args()
    return opt


def split_train_val(image_dir, trainval_percent=0.9, train_percent=0.9):
    image_filenames = os.listdir(image_dir)
    save_dir = Path(image_dir).parent.as_posix()

    num = len(image_filenames)
    index = range(num)
    trainval_num = int(num * trainval_percent)          # 训练和验证的数量
    train_num = int(trainval_num * train_percent)       # 训练的数量
    trainval = random.sample(index, trainval_num)       # 采样训练和验证集
    train = random.sample(trainval, train_num)          # 采样训练集

    ftrainval = open(save_dir + '/trainval.txt', 'w', encoding='utf-8')
    ftest = open(save_dir + '/test.txt', 'w', encoding='utf-8')
    ftrain = open(save_dir + '/train.txt', 'w', encoding='utf-8')
    fval = open(save_dir + '/val.txt', 'w', encoding='utf-8')

    for i in index:
        abs_path = Path(image_dir).joinpath(image_filenames[i]).as_posix() + '\n'    # 图片文件绝对路径
        if i in trainval:
            ftrainval.write(abs_path)
            if i in train:
                ftrain.write(abs_path)
            else:
                fval.write(abs_path)
        else:
            ftest.write(abs_path)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

    print(f"generate split txt file Success!!!, its in path {save_dir}")


if __name__ == "__main__":
    opt = parse_opt()
    split_train_val(image_dir=opt.image_dir,
                    trainval_percent=opt.trainval_percent,
                    train_percent=opt.train_percent)
