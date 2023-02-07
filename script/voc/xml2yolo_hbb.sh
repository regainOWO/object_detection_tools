#!/bin/sh
# xml2yolo2dota_obb.sh

dataset_dir=$1
image_dirname="JPEGImages"
xml_dir="${dataset_dir}/Annotations"

# 分类数据集
echo -e "split dataset: ${dataset_dir} in to train val and test"
python data/split_vocdataset.py --label-dir ${xml_dir}

# 将voc数据转成yolo，并dataset_dir目录下下生成train.txt、val.txt和test.txt
echo -e "convert voc datatype into yolo hbb datatype"
python data/convert/voc2yolo --dataset-dir ${dataset_dir}

echo -e "convert successfull!!"