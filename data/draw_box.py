"""
此脚本的内容如下
"""

import argparse
import os
import os.path as osp
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--image-dirname', type=str, default='images',
                        help='the image directory name under the dataset directory')
    parser.add_argument('--label-dirname', type=str, default='labels', help='the label directory name under the dataset directory')
    parser.add_argument('--output-dirname', type=str, default=None, help='output directory name under the dataset/plot, default is the same with --label-dirname')
    opt = parser.parse_args()
    return opt


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class DrawBbox:
    def __init__(self, class_list: list=None, thickness=2):
        """
        :param class_list:
        :param thickness:
        :param cn:
        """
        self.class_list = class_list
        self.thickness = thickness
        self.colors = colors            # 显示颜色

    def draw_single(self, img_path, label_path, output_dir):
        if not osp.exists(label_path):
            _ = open(label_path, 'w')
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        img_h, img_w = img.shape[:2]

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = str(line).strip()
                data = line.split(' ')
                label_length = len(data)
                if label_length == 5:   # id, x, y, w, h
                    # 去归一化
                    cls_id, x, y, w, h = int(data[0]), float(data[1]) * img_w, float(data[2]) * img_h, float(data[3]) * img_w, float(
                        data[4]) * img_h
                    # hbb -> poly
                    poly = [[x - 0.5 * w, y - 0.5 * h], [x + 0.5 * w, y - 0.5 * h],
                            [x + 0.5 * w, y + 0.5 * h], [x - 0.5 * w, y + 0.5 * h]]
                    poly = np.array(poly, dtype=np.int64)
                    class_name = self.class_list[cls_id]
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=self.colors(cls_id),
                                     thickness=self.thickness)
                    c1 = (int(rect[0][0]), int(rect[0][1]))
                elif label_length == 6:     # rbox
                    # 去归一化
                    cls_id, x, y, w, h, theta = int(data[0]), float(data[1]) * img_w, float(data[2]) * img_h, float(data[3]) * img_w, \
                                        float(data[4]) * img_h, int(data[5])

                    rect = ((x, y), (w, h), theta)
                    poly = np.float32(cv2.boxPoints(rect))  # 返回rect对应的四个点的值
                    poly = np.int0(poly)
                    class_name = self.class_list[cls_id]
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=self.colors(cls_id),
                                     thickness=self.thickness)
                    c1 = (int(rect[0][0]), int(rect[0][1]))

                elif label_length == 10:
                    poly = data[0:-2]
                    class_name = data[-2]
                    if class_name not in self.class_list:
                        self.class_list.append(class_name)
                    cls_id = self.class_list.index(class_name)

                    poly = list(map(float, poly))
                    poly = np.int0(poly).reshape((4, 1, 2))
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1,
                                     color=self.colors(cls_id),
                                     thickness=self.thickness)
                    c1 = np.sum(poly, axis=0)[0] / 4  # 计算中心点坐标
                    c1 = int(c1[0]), int(c1[1])
                elif label_length == 11:
                    class_name = data[-1]
                    if class_name not in self.class_list:
                        self.class_list.append(class_name)
                    cls_id = self.class_list.index(class_name)

                    conf = data[1]
                    poly = [float(num) for num in data[2:-1]]
                    poly = np.int0(poly).reshape((4, 1, 2))
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1,
                                     color=self.colors(cls_id),
                                     thickness=self.thickness)
                    c1 = np.sum(poly, axis=0)[0] / 4  # 计算中心点坐标
                    c1 = int(c1[0]), int(c1[1])
                else:
                    print("not support this data format")
                    break

                # t_size = cv2.getTextSize(class_name, 0, fontScale=1 / 4, thickness=self.thickness)[0]

                cv2.putText(img, class_name, (c1[0], c1[1] - 2), 0, 1, self.colors(cls_id), thickness=self.thickness, lineType=cv2.LINE_AA)

        img_name = Path(img_path).name
        cv2.imencode('.' + img_name.split('.')[-1], img)[1].tofile(osp.join(output_dir, img_name))

    def draw_batch(self, image_files: list, label_dir: str, output_dir: str):
        """
        对yolo和dota两种格式的数据进行画图
            0 0.8058167695999146 0.40044307708740234 0.6847715973854065 0.3581983745098114 91
            1367.196 1149.1193 1352.5274 134.8563 1893.1379 127.0375 1907.8070 1141.3006 matou 0
        @param image_files: 图像路径
        @param label_dir: 标签文件夹
        @param output_dir: 画好的图像保存路径
        @param class_list: yolo 用的是 class_id 所以需要指定其对应的类名
        @return:
        """
        os.makedirs(output_dir, exist_ok=True)
        for image_file in tqdm(image_files):
            label_name = Path(image_file).with_suffix('.txt').name
            label_path = osp.join(label_dir, label_name)
            img_path = image_file
            self.draw_single(img_path, label_path, output_dir)


def run(dataset_dir, image_dirname, label_dirname, output_dirname):
    label_dir = osp.join(dataset_dir, label_dirname)
    output_dir = osp.join(dataset_dir, 'plot', output_dirname)
    classes_file = osp.join(dataset_dir, 'classes.txt')

    image_dir = osp.join(dataset_dir, image_dirname)
    image_files = [osp.join(image_dir, x) for x in os.listdir(image_dir)]
    with open(classes_file, mode='r', encoding='utf-8') as f:
        class_list = [x.strip() for x in f.readlines()]

    draw = DrawBbox(class_list=class_list,
                    thickness=3)
    draw.draw_batch(image_files=image_files,
                    label_dir=label_dir,
                    output_dir=output_dir)


if __name__ == '__main__':
    # 可以画 yolo 格式、dota 格式、yolo 检测结果

    opt = parse_opt()
    output_dirname = opt.label_dirname if opt.output_dirname is None else opt.output_dirname
    run(dataset_dir=opt.dataset_dir,
        image_dirname=opt.image_dirname,
        label_dirname=opt.label_dirname,
        output_dirname=output_dirname)

