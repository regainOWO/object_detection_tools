"""
此脚本的主要内容:
将dota格式转成rolabellmg软件原始生成的标签格式
可以方便对标签进行再次的调整，次脚本可以配合模型预标注一起使用
"""

import argparse
import glob
import math
import os
import os.path as osp
import cv2
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np
from tqdm import tqdm

# "D:/BaiduNetdiskDownload/dota-test/labelTxt"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-dir', type=str, required=True, help='dota label dir')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='the Annotations directory name beside the dota label directory')
    opt = parser.parse_args()
    return opt


def get_file_basename(file_path: str, ext=None):
    """
    根据路径，获取文件的名称，可以修改拓展名
    Args:
        file_path (str): 文件的路径
        ext (str): 文件的拓展名
    """
    basename = osp.basename(file_path)
    if ext is not None:
        basename = osp.splitext(basename)[0] + ext
    return basename


def parse_dota(txt_file):
    with open(txt_file, mode='r', encoding='utf-8') as f:
        labels = [str(x).strip().split(' ') for x in f.readlines()]
    polys, class_names, difficults = [], [], []
    for i, label in enumerate(labels):
        class_name = label[-2]
        difficult = int(label[-1])
        difficults.append(difficult)
        class_names.append(class_name)
        polys.append(list(map(float, label[:-2])))
    return polys, class_names, difficults


def poly2xywha(polys: list, class_names, difficults):
    """
    Args:
        polys (list): 矩形的四个顶点坐标，每一行有8个值
    """
    results = []
    for poly, class_name, difficult in zip(polys, class_names, difficults):
        # c_x = (poly[0] + poly[4]) / 2
        # c_y = (poly[1] + poly[5]) / 2
        # w = math.sqrt((poly[0] - poly[2]) ** 2 +
        #     (poly[1] - poly[3]) ** 2)

        # h = math.sqrt((poly[4] - poly[2]) ** 2 +
        #     (poly[5] - poly[3]) ** 2)
        (cx, cy), (w, h), angle = cv2.minAreaRect(np.array(poly, dtype=int).reshape(4, 2))
        angle = angle * math.pi / 180.0
        result = {
            'cx': round(cx, 4),
            'cy': round(cy, 4),
            'w': round(w, 4),
            'h': round(h, 4),
            'angle': round(angle, 4),
            'name': class_name,
            'difficult': difficult
        }
        results.append(result)
    return results


def format_save_xml_file(root, xml_file, indent="\t", newl="\n", encoding="utf-8"):
    """
    格式化输出xml文件
    reference: https://blog.csdn.net/qq_29007291/article/details/106666001
    Args:
        root (Element):
        xml_file (str): xml文件路径
    """
    raw_text = ET.tostring(root)
    dom = minidom.parseString(raw_text)
    with open(xml_file, 'w', encoding=encoding) as f:
        dom.writexml(f, "", indent, newl, encoding)


def save(rboxes, filename, output_dir):
    """
    保存成xml文件
    Args:
        rboxes (list): xywha格式的box
        filename (str): 文件名
        ouput_dir (str): 输出的文件夹路径
    """
    dst_file = osp.join(output_dir, filename + '.xml')
    root = ET.Element('annotation')
    r_folder = ET.SubElement(root, 'folder')
    r_folder.text = "Unknown"
    r_filename = ET.SubElement(root, 'filename')
    r_filename.text = filename
    r_path = ET.SubElement(root, 'path')
    r_path.text = "Unknown"
    
    r_source = ET.SubElement(root, 'source')
    rr_database = ET.SubElement(r_source, 'database')
    rr_database.text = 'Unknown'

    r_size = ET.SubElement(root, 'size')
    rr_width = ET.SubElement(r_size, 'width')
    rr_width.text = 'Unknown'
    rr_height = ET.SubElement(r_size, 'height')
    rr_height.text = 'Unknown'
    rr_depth = ET.SubElement(r_size, 'depth')
    rr_depth.text = 'Unknown'

    r_segmented = ET.SubElement(root, 'segmented')
    r_segmented.text = str(0)

    for rbox in rboxes:
        r_obj = ET.SubElement(root, 'object')
        
        rr_type = ET.SubElement(r_obj, 'type')
        rr_type.text = "robndbox"
        rr_name = ET.SubElement(r_obj, 'name')
        rr_name.text = str(rbox['name'])
        rr_pose = ET.SubElement(r_obj, 'pose')
        rr_pose.text = 'Unspecified'
        rr_truncated = ET.SubElement(r_obj, 'truncated')
        rr_truncated.text = str(0)
        rr_difficult = ET.SubElement(r_obj, 'difficult')
        rr_difficult.text = str(rbox['difficult'])

        rr_robndbox = ET.SubElement(r_obj, 'robndbox')
        rrr_cx = ET.SubElement(rr_robndbox, 'cx')
        rrr_cy = ET.SubElement(rr_robndbox, 'cy')
        rrr_w = ET.SubElement(rr_robndbox, 'w')
        rrr_h = ET.SubElement(rr_robndbox, 'h')
        rrr_angle = ET.SubElement(rr_robndbox, 'angle')
        rrr_cx.text = str(rbox['cx'])
        rrr_cy.text = str(rbox['cy'])
        rrr_w.text = str(rbox['w'])
        rrr_h.text = str(rbox['h'])
        rrr_angle.text = str(rbox['angle'])

    format_save_xml_file(root, dst_file)


def run(label_dir, output_dir):
    assert osp.isdir(label_dir), "文件夹路径: " + label_dir + "不存在"
    label_files = glob.glob(osp.join(label_dir, '*.txt'))
    assert len(label_files) > 0, "路径: " + label_dir + "下没有.txt文件，请确保输入的标签文件夹路径时正确的"
    # 创建输出的文件夹
    if output_dir is None:
        output_dir = osp.join(osp.dirname(osp.abspath(label_dir)), "Annotations")
    os.makedirs(output_dir, exist_ok=True)
    for label_file in tqdm(label_files, total=len(label_files)):
        polys, class_names, difficults = parse_dota(label_file)
        rboxes = poly2xywha(polys, class_names, difficults)
        filename = osp.splitext(osp.basename(label_file))[0]
        save(rboxes, filename, output_dir)


if __name__ == "__main__":
    opt = parse_opt()
    run(label_dir=opt.label_dir,
        output_dir=opt.output_dir)