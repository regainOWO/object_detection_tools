"""
此脚本的内容如下:
将box标签画到图像中，支持的box标签格式包括，yolo，带角度的yolo，dota
输入参数
--image-path: 图片的路径，可以是文件夹或文件
--label-path: 标签的路径，可以是文件夹或文件
图片的路径和标签的路径必须同时为文件路径或者同时为文件夹路径，若图片和标签的路径都为文件夹路，文件夹下图片和标签的名称必须保持一致
--separator: 标签文件中, 每行代表一个四边形框的数据，其数据之间的分隔符号，一般默认为一个空格字符，有些的数据集为逗号字符
"""

import argparse
import logging
import os
import os.path as osp
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger("basic")

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True,
                        help='the image path')
    parser.add_argument('--label-path', type=str, required=True, help='the label path')
    parser.add_argument('--output-path', type=str, default=None, help='output path, default is under the image-path split directory')
    parser.add_argument('--separator', type=str, default=' ', help='the label separator, some label content may use comma symbol')
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


class DrawBbox:
    def __init__(self, thickness=2, label_separator=' '):
        """
        :param class_list:
        :param thickness:
        :param cn:
        """
        self.thickness = thickness
        self.colors = Colors()            # 显示颜色
        self.classes = {}
        self.label_separator = label_separator

    def draw_single(self, img_path, label_path, output_dir):
        if not osp.exists(label_path):
            _ = open(label_path, 'w')
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        img_h, img_w = img.shape[:2]

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = str(line).strip()
                data = line.split(self.label_separator)
                label_length = len(data)
                if label_length == 5:   # id, x, y, w, h
                    # 去归一化
                    cls_id, x, y, w, h = int(data[0]), float(data[1]) * img_w, float(data[2]) * img_h, float(data[3]) * img_w, float(
                        data[4]) * img_h
                    # hbb -> poly
                    poly = [[x - 0.5 * w, y - 0.5 * h], [x + 0.5 * w, y - 0.5 * h],
                            [x + 0.5 * w, y + 0.5 * h], [x - 0.5 * w, y + 0.5 * h]]
                    poly = np.array(poly, dtype=np.int64)
                    if cls_id not in self.classes.keys():
                        self.classes[cls_id] = cls_id
                    class_name = str(cls_id)
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=self.colors(cls_id),
                                     thickness=self.thickness)
                    c1 = (int(rect[0][0]), int(rect[0][1]))
                elif label_length == 6:     # rbox
                    # 去归一化
                    cls_id, x, y, w, h, theta = int(data[0]), float(data[1]) * img_w, float(data[2]) * img_h, float(data[3]) * img_w, \
                                        float(data[4]) * img_h, int(data[5])

                    rect = ((x, y), (w, h), theta)
                    poly = np.float32(cv2.boxPoints(rect))  # 返回rect对应的四个点的值
                    poly = np.intp(poly)
                    if cls_id not in self.classes.keys():
                        self.classes[cls_id] = cls_id
                    class_name = str(cls_id)
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=self.colors(cls_id),
                                     thickness=self.thickness)
                    c1 = (int(rect[0][0]), int(rect[0][1]))

                elif label_length == 10:
                    poly = data[0:-2]
                    class_name = data[-2]
                    if class_name not in self.classes.values():
                        length = len(self.classes.values())
                        self.classes[length + 1] = class_name
                    cls_id = list(self.classes.values()).index(class_name)
                    # class_name = str(cls_id)

                    poly = list(map(float, poly))
                    poly = np.intp(poly).reshape((4, 1, 2))
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1,
                                     color=self.colors(cls_id),
                                     thickness=self.thickness)
                    c1 = np.sum(poly, axis=0)[0] / 4  # 计算中心点坐标
                    c1 = int(c1[0]), int(c1[1])
                elif label_length == 11:
                    class_name = data[-1]
                    if class_name not in self.classes.values():
                        length = len(self.classes.values())
                        self.classes[length + 1] = class_name
                    cls_id = list(self.classes.values()).index(class_name)
                    # class_name = str(cls_id)

                    conf = data[1]
                    poly = [float(num) for num in data[2:-1]]
                    poly = np.intp(poly).reshape((4, 1, 2))
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1,
                                     color=self.colors(cls_id),
                                     thickness=self.thickness)
                    c1 = np.sum(poly, axis=0)[0] / 4  # 计算中心点坐标
                    c1 = int(c1[0]), int(c1[1])
                else:
                    logger.info("not support this data format")
                    break

                # t_size = cv2.getTextSize(class_name, 0, fontScale=1 / 4, thickness=self.thickness)[0]

                cv2.putText(img, class_name, (c1[0], c1[1] - 2), 0, 1, self.colors(cls_id), thickness=self.thickness, lineType=cv2.LINE_AA)

        img_name = Path(img_path).name
        cv2.imencode('.' + img_name.split('.')[-1], img)[1].tofile(osp.join(output_dir, img_name))

    def draw_batch(self, image_files: list, label_dir: str, output_dir: str):
        """
        对yolo和dota三种格式的数据进行画图
            0 0.8058167695999146 0.40044307708740234 0.6847715973854065 0.3581983745098114
            0 0.8058167695999146 0.40044307708740234 0.6847715973854065 0.3581983745098114 91
            1367.196 1149.1193 1352.5274 134.8563 1893.1379 127.0375 1907.8070 1141.3006 matou 0
        @param image_files: 图像路径
        @param label_dir: 标签文件夹
        @param output_dir: 画好的图像保存路径
        @param class_list: yolo 用的是 class_id 所以需要指定其对应的类名
        @return:
        """
        os.makedirs(output_dir, exist_ok=True)
        pbar = tqdm(image_files, total=len(image_files))
        for image_file in pbar:
            pbar.desc = f"plot image file: {osp.abspath(image_file)}"
            label_name = Path(image_file).with_suffix('.txt').name
            label_path = osp.join(label_dir, label_name)
            img_path = image_file
            self.draw_single(img_path, label_path, output_dir)
        print(f"绘制完成! 图片文件保存在目录: {osp.abspath(output_dir)}")


def run(image_path, label_path, output_path, label_separator=' '):
    assert osp.exists(image_path), f'image path: {image_path} do not exist'
    assert osp.exists(label_path), f'label path: {label_path} do not exist'
    # 处理图片路径
    if osp.isdir(image_path):
        image_dir = image_path
        image_files = [osp.join(image_dir, x) for x in os.listdir(image_dir) if Path(x).suffix.rsplit('.')[-1] in IMG_FORMATS]
    else:
        image_dir = Path(image_path).parent.as_posix()
        image_files = [image_path]

    # 处理标签路径
    if osp.isdir(label_path):
        label_dir = label_path
    else:
        assert len(image_files) == 1, 'image path is a dir with multiple images, but you set label path is a file'
        label_dir = Path(label_path).parent.as_posix()
        image_file_stem = Path(image_files[0]).stem
        if Path(label_path).stem != image_file_stem:
            logger.info("label file stem name is not equal to image file, try rename the label file....")
            dst_label_path = osp.join(label_dir, image_file_stem + '.txt')
            os.rename(label_path, dst_label_path)

    # 指定输出的路径，设一个默认值
    if output_path is None:
        output_path = image_dir + '_plot'
    else:
        output_path = osp.join(output_path, osp.basename(image_dir) + '_plot')

    draw = DrawBbox(thickness=2, label_separator=label_separator)
    draw.draw_batch(image_files=image_files,
                    label_dir=label_dir,
                    output_dir=output_path)


if __name__ == '__main__':
    # 可以画 yolo 格式、dota 格式、yolo 检测结果

    opt = parse_opt()
    run(image_path=opt.image_path,
        label_path=opt.label_path,
        output_path=opt.output_path,
        label_separator=opt.separator)
