"""
脚本介绍:
对key_object_final_results.csv 文件进行处理，将其中有记录的图片复制到指定的目录下
输入参数如下
1. --project-dir: 1-0-02J001-20230712 目录的路径目录结构如下

1-0-02J001-20230712
├── ProcessData
│   ├── ...
│   └── Data
│       ├── ...
│       └── Img
│           ├── ...
│           ├── Camera1
│           └── Camera2
└── key_object_final_results.csv

2. --output-dir: 结果存放路径(可以不输入，默认为--project-dir目录下的output文件夹)
"""

import argparse
import glob
import logging
import shutil
import os
import os.path as osp
import pandas as pd

from tqdm import tqdm


def get_logger(log_name):
    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler()
    ch_format = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s')
    ch.setFormatter(ch_format)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger


# logger
LOGGER = get_logger('csv2voc')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', type=str, required=True,
                        help='1-0-02J001-20230712 目录的路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出文件夹的路径，默认路径再project-dir目录下的output文件夹中')
    opt = parser.parse_args()
    return opt


def decode_csv_file(csv_file, separator=' '):
    df = pd.read_csv(csv_file)

    colunm_names = df.columns.tolist()
    process_csv_pbar = tqdm(df.iterrows(), desc=f"preprocess with the file: {csv_file}")
    img_filenames = []
    for index, row in process_csv_pbar:
        filename = row['image_name_1_close_shot']   # 取远景图名称
        img_filenames.append(filename)
    return img_filenames


def run(project_dir: str, output_dir: str = None):
    if output_dir is None:
        output_dir = osp.join(project_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    csv_file = osp.join(project_dir, 'key_object_final_results.csv')
    assert osp.isfile(csv_file), f'.csv文件: {csv_file} 不存在'

    # origin img
    img_dir_root = osp.join(project_dir, 'ProcessData', 'Data', 'Img')
    camera1_img_dir = osp.join(img_dir_root, 'Camera1')
    camera2_img_dir = osp.join(img_dir_root, 'Camera2')
    # dst img
    dst_camera1_img_dir = osp.join(output_dir, 'Camera1')
    dst_camera2_img_dir = osp.join(output_dir, 'Camera2')
    os.makedirs(dst_camera1_img_dir, exist_ok=True)
    os.makedirs(dst_camera2_img_dir, exist_ok=True)

    img_filenames = decode_csv_file(csv_file)
    process_bar = tqdm(img_filenames, total=len(img_filenames), desc='copy image file')
    for filename in process_bar:
        # camera 1
        origin_img1_file = osp.join(camera1_img_dir, filename)
        dst_img1_file = osp.join(dst_camera1_img_dir, filename)
        frame_name = str(filename).split('-')[-1]
        # camera 2
        # '040-2-*-000319'
        search_img2_list = glob.glob(camera2_img_dir + f'/*-{frame_name}')
        if len(search_img2_list) > 0:
            origin_img2_file = search_img2_list[0]
        else:
            origin_img2_file = camera2_img_dir + f'/*-{frame_name}'
        img2_filename = osp.basename(origin_img2_file)
        dst_img2_file = osp.join(dst_camera2_img_dir, img2_filename)

        # process_bar.desc = f'copy image file: {origin_img1_file}'
        # camera 1 img
        if not osp.isfile(origin_img1_file):
            LOGGER.warning(f'file: {origin_img1_file} dose not exist, it will be ignore')
        else:
            shutil.copy(origin_img1_file, dst_img1_file)
        # camera 2 img
        if not osp.isfile(origin_img2_file):
            LOGGER.warning(f'file: {origin_img2_file} dose not exist, it will be ignore')
        else:
            shutil.copy(origin_img2_file, dst_img2_file)
        
    # csv file
    dst_csv_file = osp.join(output_dir, osp.basename(csv_file))
    shutil.copy(csv_file, dst_csv_file)


if __name__ == "__main__":
    opt = parse_opt()
    run(project_dir=opt.project_dir,
        output_dir=opt.output_dir)
    