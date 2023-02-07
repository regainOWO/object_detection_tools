#!/bin/sh
# split_image.sh
# E:/GithubProjects/datasets/XiAnSample/08-AirportElement/油库/split

dataset_dir=$1
image_dirname="images"
label_dirname="labelTxt"
output_path="${dataset_dir}/split"

# 将dota格式转成yolo hbb格式
echo -e "split dota image..."
python data/imgsplit/gap_split.py --dataset-dir ${dataset_dir} \
                                  --gap 200 \
                                  --subsize 1024 \
                                  --scale 1 \
                                  --image-dirname ${image_dirname} \
                                  --label_dirname ${label_dirname} \
                                  --output-path ${output_path}