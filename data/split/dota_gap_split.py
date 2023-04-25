"""
此脚本主要是对较大的dota图像进行切割到指定的尺寸
1. 切割的结果放在一起形成原图，切割图像中会有重叠，可以用参数gap进行指定，默认值为200。这样就减少了检测物体被切割成两半的概率，从而提高了数据集的准确度
2. 若图像不足切割的尺寸，会对图进行padding，在右下角进行填充，这样lable的坐标就不用进行更改了。
3. 使用多进程进行处理，大图切割成小图，不同的图像可以进行同样的操作，互不干扰。每个进程分别处理不同的图片，加快处理速度

参考资料:
DOTA遥感数据集以及相关工具DOTA_devkit的整理(踩坑记录): https://zhuanlan.zhihu.com/p/355862906
代码参考: https://github.com/hukaixuan19970627/DOTA_devkit_YOLO
"""

import argparse
from copy import deepcopy
import logging
import math
import os
import os.path as osp
import re
import shutil
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import numpy as np
import shapely.geometry as shgeo
from tqdm import tqdm

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--scale2subsize', action='store_true',
                        help='do not split, just scale img to subsize')
    parser.add_argument('--gap', type=int, default=200,
                        help='split gap size, default is 20')
    parser.add_argument('--subsize', type=int, default=1024,
                        help='spilt image result pixel size [subsize, subsize], its a square, default is 1024')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='scale image before spilt image result, default is 1')
    parser.add_argument('--image-dirname', type=str, default='images',
                        help='the image directory name under the dataset directory, default is images')
    parser.add_argument('--label-dirname', type=str, default='labelTxt',
                        help='the dota label directory name under the dataset directory, default is labelTxt')
    parser.add_argument('--output-path', type=str, default='split',
                        help='output path, default is split directory under the your workingdirectory')
    parser.add_argument('--save-ext', type=str, default='.png',
                        help='split image save suffix, default is .png')
    parser.add_argument('--nosave-empty', action='store_true',
                        help='split image may do no have label, it will not save image and label')
    parser.add_argument('--workers', type=int, default=4, help='max split workers process num')
    opt = parser.parse_args()
    return opt


def get_path_basename(path):
    p = Path(path)
    return p.name.split(p.suffix)[0]


def poly_flatten(poly):
    out_poly = []
    for point in poly:
        out_poly.append(point[0])
        out_poly.append(point[1])
    return out_poly


def parse_dota_label(label_file):
    with open(label_file, mode='r', encoding='utf-8') as f:
        lines = [x.strip() for x in f.readlines() if re.match('(\d*[.]\d\s){8}', x)]

    objects = []
    for line in lines:
        object = {}
        data = line.split(' ')
        poly = [
            (float(data[0]), float(data[1])),
            (float(data[2]), float(data[3])),
            (float(data[4]), float(data[5])),
            (float(data[6]), float(data[7])),
        ]
        if len(data) >= 9:
            object['name'] = data[8]
        if len(data) >= 10:
            object['difficult'] = data[9]
        # 计算poly的面积
        geo_poly = shgeo.Polygon(poly)
        object['area'] = geo_poly.area
        # 将poly展平
        poly = poly_flatten(poly)
        object['poly'] = poly
        objects.append(object)
    return objects


def get_best_point_order(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]


def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


class GapSplit:
    def __init__(self,
                 dataset_dir,
                 output_path,
                 image_dirname='images',
                 label_dirname='labelTxt',
                 scale2subsize=False,
                 gap=100,
                 subsize=1024,
                 thresh=0.7,
                 best_order=True,
                 padding=True,
                 ext='.png',
                 save_empty=True,
                 num_process=1):
        """
        :param dataset_dir: dataset_dir path for dota data
        :param output_path: output dataset path for dota data,
        the outputpath have the subdirectory, 'images' and 'labelTxt'
        :param scale2subsize: do not split image, scale origin image to subsize
        :param image_dirname: image dir name under the dataset dir
        :param label_dirname: label dir name under the dataset dir
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for save image format
        :param save_empty: split image may not have label, if true, it will save empty label and image file
        """
        self.dataset_dir = dataset_dir
        self.output_path = output_path
        self.scale2subsize = scale2subsize
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.image_dir = osp.join(self.dataset_dir, image_dirname)
        self.label_dir = osp.join(self.dataset_dir, label_dirname)
        self.o_image_dir = osp.join(self.output_path, 'images')
        self.o_label_dir = osp.join(self.output_path, 'labelTxt')
        self.best_order = best_order
        self.padding = padding
        self.ext = ext
        self.save_empty = save_empty
        self.num_process = min(num_process, NUM_THREADS)

        # 删除之前切割的文件
        if osp.exists(self.output_path):
            shutil.rmtree(self.output_path)
        # 创建文件夹
        os.makedirs(self.o_image_dir)
        os.makedirs(self.o_label_dir)

    def poly_origin2sub(self, left, top, poly):
        """将原图的标签box映射到切割图上的标签box"""
        poly_in_sub = np.zeros(len(poly))
        for i in range(len(poly) // 2):
            x = max(poly[i * 2] - left, 1)
            y = max(poly[i * 2 + 1] - top, 1)
            poly_in_sub[i * 2] = x if x < self.subsize else self.subsize        # 将坐标限制在图片中
            poly_in_sub[i * 2 + 1] = y if y < self.subsize else self.subsize
        return poly_in_sub

    def poly_ioa(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)  # poly1和poly2的交集
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def save_sub_image(self, img, sub_img_name, left, top):
        """保存切割的图片"""
        # sub_img = deepcopy(img[top: (top + self.subsize), left: (left + self.subsize), :])
        sub_img = deepcopy(img[top: (top + self.subsize), left: (left + self.subsize)])
        o_image_file = os.path.join(self.o_image_dir, sub_img_name + self.ext)
        h, w = sub_img.shape[:2]
        if self.padding and [h, w] != [self.subsize, self.subsize]:
            o_img = cv2.resize(np.zeros_like(sub_img), (self.subsize, self.subsize))
            # o_img[0:h, 0:w, :] = sub_img
            o_img[0:h, 0:w] = sub_img
        else:
            o_img = sub_img

        # cv2.imwrite(o_image_file, o_img)
        cv2.imencode(self.ext, o_img)[1].tofile(o_image_file)

    def get_fix_poly4_from_poly5(self, poly):
        """
        5边形中切出面积最大的4边形
        方法：去掉最短边的两个点，将这两个点的重点作为第4个点，其他三个点不变
        """
        # 分别计算5条边的长度
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(len(poly) // 2 - 1)]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]  # 最短边的索引值
        # 最短边两个点的索引组成一个列表
        short_startpoint_i = [(pos * 2) % 10, (pos * 2 + 1) % 10]
        short_endpoint_i = [((pos + 1) * 2) % 10, ((pos + 1) * 2) % 10 + 1]
        # 取中值，得到新的点
        combine_point_x = (poly[short_startpoint_i[0]] + poly[short_endpoint_i[0]]) / 2
        combine_point_y = (poly[short_startpoint_i[1]] + poly[short_endpoint_i[1]]) / 2
        # 将最短边的两点删除
        if short_endpoint_i[0] != 0:
            del poly[short_startpoint_i[0]:(short_endpoint_i[1] + 1)]       # 一起删除，避免index重新调整出错
        else:
            del poly[short_startpoint_i[0]:(short_startpoint_i[1] + 1)]     # 先删除后面的
            del poly[short_endpoint_i[0]:(short_endpoint_i[1] + 1)]         # 在删除前面的
        # 将两个点插入进去
        poly.insert(short_startpoint_i[0], combine_point_x)
        poly.insert(short_startpoint_i[1], combine_point_y)
        return poly

    def save_sub_files(self, img, objects, sub_img_name, left, top, right, bottom):
        # 新标签文件的路径
        o_label_file = os.path.join(self.o_label_dir, sub_img_name + '.txt')

        geo_img_poly = shgeo.Polygon([
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom)
        ])

        content_lines = []  # 新标签文件中每行的数据
        for obj in objects:
            geo_gt_poly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
            if geo_gt_poly.area <= 0:
                continue
            inter_poly, ioa = self.poly_ioa(geo_gt_poly, geo_img_poly)     # 计算交集 和 交集占geo_gt_poly的比值（判断这个box是否完全在image中)

            if ioa == 1: # gt完全在image
                poly_in_sub = self.poly_origin2sub(left, top, obj['poly'])
                content_line = ' '.join(["%.1f" % x for x in poly_in_sub])
                content_line = content_line + ' ' + obj['name'] + ' ' + obj['difficult'] + '\n'
                content_lines.append(content_line)
            elif ioa > 0:    # gt部分在image中
                inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                out_poly = list(inter_poly.exterior.coords)[0: -1]
                overlap_point_num = len(out_poly)
                out_poly = poly_flatten(out_poly)

                if overlap_point_num < 4:   # gt与image重叠的多边形的点个数小于等于三，即重叠的多边形为三角形，gt只有一个小角在image中
                    continue            # 过滤掉这个gt
                if overlap_point_num == 5:  # gt与image重叠的为5边形，表示大部分框都在image里面，将5个点整合成4个点
                    out_poly = self.get_fix_poly4_from_poly5(out_poly)
                elif overlap_point_num > 5:
                    """
                        if the cut instance is a polygon with points more than 5, we do not handle it currently
                    """
                    continue
                # 其中gt与image重叠为4边形的，直接取
                if self.best_order:
                    out_poly = get_best_point_order(out_poly, obj['poly'])

                poly_in_sub = self.poly_origin2sub(left, top, out_poly)

                content_line = ' '.join(["%.1f" % x for x in poly_in_sub])
                # 根据阈值，设置标签识别的难易度
                if ioa > self.thresh:
                    content_line = content_line + ' ' + obj['name'] + ' ' + obj['difficult'] + '\n'
                else:
                    ## if the left part is too small, label as '2'
                    content_line = content_line + ' ' + obj['name'] + ' ' + '2' + '\n'
                content_lines.append(content_line)
        # if len(content_lines):      
        if self.save_empty or len(content_lines):   # 当前子图有标签时或要保存空图，才保存标签和图片文件
            with open(o_label_file, mode='w', encoding='utf-8') as f:
                f.writelines(content_lines)
            self.save_sub_image(img, sub_img_name, left, top)

    def get_fix_windowsize(self, point, interval, max_length):
        """
        :param point: 当前所处的位置
        :param interval: window的尺寸
        :param max_length: 图片的尺寸
        """
        if point + interval < max_length:
            return point, interval
        else:
            d = max_length - interval
            if d >= 0:                  # 起始点位于图中，但往后切的图不满足尺度达不到interval
                return point, d
            else:                       # 其实点位于图的最左侧，但往后切的图不满足尺度达不到interval
                return 0, max_length

    def slide_cutout(self, width, height, window_size, gap, mode=0):
        """
        对尺寸进行窗口口切割，并返回窗口的左上角坐标和尺寸
        @param: width: 原图的宽度
        @param: height: 原图的高度
        @param: window_size: 子图的尺寸
        @param: gap: 裁剪的重叠像素尺度

        return: (left, top), (width, height)
        """
        if isinstance(window_size, int):
            wds_w = wds_h = window_size
        else:
            assert isinstance(window_size, (list, tuple))
            wds_w, wds_h = window_size[:2]
        stride_x = wds_w - gap
        stride_y = wds_h - gap
        # 按照从左往右，从上往下，逐行扫描
        # 切割尺寸保持不变，当切割到图外面的时候，滑块的右下角提到图片的边界，若此时左上角为超出图片边界，则取window_size，若超出边界滑块左上角提到图片边界，右下角为图片的边界，此时就需要padding了。
        for y in range(0, height, stride_y):
            for x in range(0, width, stride_x):
                x, real_wds_w = self.get_fix_windowsize(x, wds_w, width)
                y, real_wds_h = self.get_fix_windowsize(y, wds_h, height)
                yield (x, y), (real_wds_w, real_wds_h)

    def single_split(self, args):
        image_file, rate = args
        """
        split a single image and ground truth
        :param image_file: image file path
        :param rate: the resize scale for the image
        :return:
        """
        # img = cv2.imread(image_file)
        img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), -1)
        if len(img.shape) == 2:     # 灰度图只有两个通道，转成三个通道
            img = np.expand_dims(img, axis=2)
            img = np.concatenate((img, img, img), axis=-1)
        name = get_path_basename(image_file)

        if img is None:
            logging.warning(f"file: {image_file} is broken!!! we cannot load image")
            return
        label_file = osp.join(self.label_dir, name + '.txt')
        objects = parse_dota_label(label_file)
        if self.scale2subsize:
            h, w = img.shape[:2]
            rate = self.subsize / min(h, w)
        # 根据rate对标签和图片进行缩放
        for obj in objects:
            obj['poly'] = list(map(lambda x: rate * x, obj['poly']))

        if rate != 1:
            r_img = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            r_img = img
        # 保存文件的名称
        outfile_basename = name + '__' + str(rate) + '__'
        height, width = r_img.shape[:2]
        # 得到切割点左上角的坐标，以及切割尺寸。
        for (left, top), (subsize_w, subsize_h) in self.slide_cutout(width, height, self.subsize, gap=self.gap):
            sub_image_name = outfile_basename + str(left) + '___' + str(top)
            right = left + subsize_w
            bottom = top + subsize_h
            self.save_sub_files(r_img, objects, sub_image_name, left, top, right, bottom)
        return name

    def split_image(self, rate):
        """
        :param rate: resize rate before cut
        """
        image_files = [osp.join(self.image_dir, x) for x in os.listdir(self.image_dir) if x.rsplit('.')[-1] in IMG_FORMATS]
        desc = f"Spliting images in dataset path: {self.dataset_dir}......"
        if self.num_process > 1:

            with Pool(self.num_process) as pool:
                pbar = tqdm(pool.imap(self.single_split, zip(image_files, repeat(rate))),
                            desc=desc, total=len(image_files))
                for name in pbar:
                    pbar.desc = f"Spliting image: {name} ......"
            pbar.close()

        else:
            for image_file in tqdm(image_files, total=len(image_files), desc=desc):
                self.single_split([image_file, rate])


def run(dataset_dir, scale2subsize, gap, subsize, scale, image_dirname, label_dirname, output_path, ext, save_empty=True, workers=1):
    # example usage of ImgSplit
    split = GapSplit(dataset_dir=dataset_dir,
                     output_path=output_path,
                     scale2subsize=scale2subsize,
                     image_dirname=image_dirname,
                     label_dirname=label_dirname,
                     gap=gap,
                     subsize=subsize,
                     ext=ext,
                     save_empty=save_empty,
                     num_process=workers)
    split.split_image(scale)
    print("Split images Success!!!")


if __name__ == '__main__':
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        scale2subsize=opt.scale2subsize,
        gap=opt.gap,
        subsize=opt.subsize,
        scale=opt.scale,
        image_dirname=opt.image_dirname,
        label_dirname=opt.label_dirname,
        output_path=opt.output_path,
        ext=opt.save_ext,
        save_empty=not opt.nosave_empty,
        workers=opt.workers)