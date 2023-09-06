"""
此脚本的主要内容:
将dota格式转成.shp矢量文件

所需的参数如下
--image-dir 图片文件夹的路径
--label-dir dota标签的文件夹路径
--classes-file dota标签所有的类别信息.txt文件路径
--output-dir 输出的路径, 默认值为图片文件夹的同级的shp文件夹

最终输出的文件夹的目录如下
.
├─aa
│  ├─aa.dbf
│  ├─aa.prj
│  ├─aa.shp
│  ├─aa.shx
│  └─aa.jpg
├─bb
│  ├─bb.dbf
│  ├─bb.prj
│  ├─bb.shp
│  ├─bb.shx
│  └─bb.png
├─...
└─gg
    ├─...
    └─gg.jpg

"""

import argparse
import os
import os.path as osp
import shutil

from PIL.Image import init, EXTENSION
from osgeo import ogr, gdal, osr
from tqdm import tqdm

if not len(EXTENSION):
    init()


"""
D:/BaiduNetdiskDownload/dota-test/images
D:/BaiduNetdiskDownload/dota-test/labels
D:/BaiduNetdiskDownload/dota-test/classes.txt
"""


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True, help='image file directory')
    parser.add_argument('--label-dir', type=str, required=True,
                        help='label file directory')
    parser.add_argument('--classes-file', type=str, required=True,
                        help='classes.txt file path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='the shp directory name beside the label directory')
    opt = parser.parse_args()
    return opt


def parser_dota(txt_file):
    with open(txt_file, mode='r', encoding='utf-8') as f:
        labels = [str(x).strip().split(' ') for x in f.readlines()]
    result = []
    for label in labels:
        class_name = label[-2]
        difficult = int(label[-1])
        data = {
            "poly": list(map(float, label[:-2])),
            "name": class_name,
            "difficult": difficult
        }
        result.append(data)
    return result


def coordinates_pixle2geographic(geo_transform, px, py):
    """
    将影像上的像素坐标转为地理坐标
    Args:
        geo_transform (list): 坐标系转换的矩阵
        px (int): 像素坐标x
        py (int): 像素坐标y
    Returns:
        geo_x (float): 地理坐标x
        geo_y (float): 地理坐标y
    """
    geo_tl_x = geo_transform[0]              # 0
    geo_tl_y = geo_transform[3]              # 1024
    pixel_width = geo_transform[1]           # 1
    pixel_height = geo_transform[5]          # -1
    rotate1 = geo_transform[2]               # 0
    rotate2 = geo_transform[4]               # 0

    geo_x = geo_tl_x + px * pixel_width + py * rotate1
    geo_y = geo_tl_y + px * rotate2 + py * pixel_height

    return geo_x, geo_y


def dota2shp(image_file, label_data, shp_file, class_names_list: list):
    """
    Args:
        image_file (str): 输入图像的路径，用来获取尺寸数据
        label_data (list): 标签数据
        shp_file (str): 保存结果存放的路径
        class_names_list (list): 类别列表
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if osp.exists(shp_file):
        driver.DeleteDataSource(shp_file)       # 删除.shp文件
    # 获取图像的尺寸关系
    image = gdal.Open(image_file)
    image_width = image.RasterXSize
    image_height = image.RasterYSize
    geo_transform = image.GetGeoTransform()
    proj = osr.SpatialReference()
    if geo_transform == (0, 1, 0, 0, 0, 1):  # 这是默认投影，会导致上下翻转
        proj.ImportFromEPSG(4326)
        geo_transform = (0, 1, 0, image_height, 0, -1)
    else:
        # 如果图像有投影，那就从图像中获取投影
        proj.ImportFromWkt(image.GetProjection())
    # 创建数据源
    data_source = driver.CreateDataSource(shp_file)

    # 创建图层
    layer = data_source.CreateLayer("object", srs=proj, geom_type=ogr.wkbPolygon)

    # 添加字段
    field_id = ogr.FieldDefn('class_id', ogr.OFTInteger)  # 字段名，数据格式（整型
    field_name = ogr.FieldDefn('class_name', ogr.OFTString)
    layer.CreateField(field_id)
    layer.CreateField(field_name)

    for data in label_data:
        poly = data['poly']
        class_name = data['name']
        # 创建环
        ring = ogr.Geometry(ogr.wkbLinearRing)
        geo_x_0, geo_y_0 = coordinates_pixle2geographic(geo_transform, poly[0], poly[1])    # 其实坐标点
        for i in range(0, len(poly), 2):
            geo_x, geo_y = coordinates_pixle2geographic(geo_transform, poly[i], poly[i + 1])
            ring.AddPoint(geo_x, geo_y)
        ring.AddPoint(geo_x_0, geo_y_0)     # 5 point
        # 创建多边形
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        # 创建要素
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField('class_id', class_names_list.index(class_name) + 1)     # 设置属性字段
        feature.SetField('class_name', class_name)
        feature.SetGeometry(polygon)                  # 设置集合
        layer.CreateFeature(feature)
        feature = None
    data_source = None


def run(image_dir, label_dir, output_dir, classes_file):
    assert osp.isdir(image_dir), f'图片文件夹不存在: {image_dir}'
    assert osp.isdir(label_dir), f'标签文件夹不存在: {label_dir}'
    assert osp.isfile(classes_file), f'类别文件不存在: {classes_file}'

    with open(classes_file, mode='r', encoding='utf-8') as f:
        class_names = [str(x).strip() for x in f.readlines()]

    image_files = [osp.join(image_dir, x) for x in os.listdir(image_dir) if osp.splitext(x)[-1] in EXTENSION]
    if output_dir is None:
        output_dir = osp.join(osp.dirname(osp.abspath(image_dir)), 'shp')
    os.makedirs(output_dir, exist_ok=True)
    for image_file in tqdm(image_files, total=len(image_files)):
        image_basename = osp.basename(image_file)
        image_filename = osp.splitext(image_basename)[0]
        label_file = osp.join(label_dir, image_filename + '.txt')
        if not osp.isfile(label_file):
            continue
        shp_save_dir = osp.join(output_dir, image_filename)
        os.makedirs(shp_save_dir, exist_ok=True)
        shp_file = osp.join(shp_save_dir, image_filename + '.shp')
        label_data = parser_dota(label_file)
        # 生成shp文件
        dota2shp(image_file, label_data, shp_file, class_names)
        # 移动图片文件
        dst_image_file = osp.join(shp_save_dir, image_basename)
        shutil.copy(image_file, dst_image_file)


if __name__ == '__main__':
    opt = parse_opt()
    run(image_dir=opt.image_dir,
        label_dir=opt.label_dir,
        output_dir=opt.output_dir,
        classes_file=opt.classes_file)
