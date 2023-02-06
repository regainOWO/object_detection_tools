#!/bin/sh
# xml2yolo2dota_hbb.sh
# E:/GithubProjects/datasets/XiAnSample/05-PointElement/水塔
dataset_dir=$1
image_dirname=$2
xml_dir="${dataset_dir}/Annotations"

# 分类数据集
echo -e "split dataset: ${dataset_dir} in to train val and test"
python data/split_data.py --xml-dir ${xml_dir}

# 将voc数据转成yolo obb，并dataset_dir目录下下生成train.txt、val.txt和test.txt
echo -e "convert voc datatype into yolo obb datatype"
python data/convert/voc2yolo.py --dataset-dir ${dataset_dir} --obb
# 将yolo obb格式转成 dota格式
echo -e "convert yolo obb datatype into dota datatype"
python data/convert/le180yolo2dota.py --dataset-dir ${dataset_dir} --image-dirname ${image_dirname}
# 将dota格式转成yolo hbb格式
echo -e "convert dota datatype into yolo hbb datatype"
python data/convert/dota2yolo.py --dataset-dir ${dataset_dir} --image-dirname ${image_dirname}

echo -e "convert successfull!!"