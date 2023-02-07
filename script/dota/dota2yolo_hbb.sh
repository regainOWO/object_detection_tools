#!/bin/sh
# dota2yolo_hbb.sh
# E:/GithubProjects/datasets/XiAnSample/08-AirportElement/油库/split

dataset_dir=$1
image_dirname="images"
label_dir="labelTxt"

# 将dota格式转成yolo hbb格式
echo -e "convert dota datatype into yolo hbb datatype"
python data/convert/dota2yolo.py --dataset-dir ${dataset_dir} --image-dirname ${image_dirname}