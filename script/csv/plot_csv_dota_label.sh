#!/bin/sh
# plot_csv_dota_label.sh
# 脚本： 将dota的csv文件转成dota的标签格式，并将标签画到原图上，用于测试集的展示

CSV_PATH=$1
TEST_PATH=$2
SEPARATOR=$3

# 进入tools目录
TOOLS_DIR=$(cd "$(dirname "$0")";pwd)
TOOLS_DIR=$(pwd)

# csv2labels
python ${TOOLS_DIR}/convert/file/csv2dota.py --csv-path ${CSV_PATH} --output-dir ${TEST_PATH} --separator ${SEPARATOR}
# 画图
python ${TOOLS_DIR}/plot/draw_box_label.py  --image-path ${TEST_PATH}/images --label-path ${TEST_PATH}/labels --separator ${SEPARATOR}