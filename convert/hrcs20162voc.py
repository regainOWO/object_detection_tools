"""
此脚本的主要内容
1. 将HRCS-2016的数据格式转换成voc格式
2. 标签名称全转成ship

使用方法，需要输入三个参数：
--ann-dir: HRCS-2016的xml标签文件存放的路径
--output-dir: pascal voc格式输出的路径，默认路径为HRCS-2016标签文件夹的同级 Annotations_voc目录
--noempty: 是否保存空标签文件，默认是保存的
"""

import argparse
import glob
import logging
from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import os.path as osp

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
LOGGER = get_logger('hrcs20162voc')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-dir', type=str, required=True,
                        help='input HRCS-2016 annotations dir')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output pascal voc annotation dir, default is beside the ann-dir directory called Annotations_voc')
    parser.add_argument('--noempty', action='store_true',
                        help='if or not save empty label, default is not save')
    opt = parser.parse_args()
    return opt


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


def str_list_to_txtfile(txt_file: str, data: list):
    """
    将字符列表保存成.txt文件
    Args:
        txt_file (str): .txt文件路径
        data (list): 字符数组
    """
    with open(txt_file, 'w', encoding='utf-8') as f:
        contents = [x + '\n' for x in data]
        f.writelines(contents)


def generate_voc_xml_file(xml_file, image_file, img_shape, annotations: list):
    """
    保存成pascal voc 格式.xml文件
    Args:
        xml_file (str): 标签保存路径
        image_file (str): 图片路径
        img_shape (str): 图片宽高和维度
        annotations (list): 标签内容
    """
    image_file = image_file.replace('\\', '/')
    width, height, channel = img_shape     # 获取图片的宽高

    root = ET.Element('annotation')
    r_folder = ET.SubElement(root, 'folder')
    r_folder.text = 'Unknown'

    r_filename = ET.SubElement(root, 'filename')
    r_filename.text = os.path.basename(image_file)
    r_path = ET.SubElement(root, 'path')
    r_path.text = 'Unknown'

    r_source = ET.SubElement(root, 'source')
    rr_database = ET.SubElement(r_source, 'database')
    rr_database.text = 'Unknown'

    r_size = ET.SubElement(root, 'size')
    rr_width = ET.SubElement(r_size, 'width')
    rr_width.text = str(width)
    rr_height = ET.SubElement(r_size, 'height')
    rr_height.text = str(height)
    rr_depth = ET.SubElement(r_size, 'depth')
    rr_depth.text = str(channel)

    r_segmented = ET.SubElement(root, 'segmented')
    r_segmented.text = str(0)

    for ann in annotations:
        r_obj = ET.SubElement(root, 'object')

        rr_name = ET.SubElement(r_obj, 'name')
        rr_name.text = str(ann['name'])
        rr_pose = ET.SubElement(r_obj, 'pose')
        rr_pose.text = 'Unspecified'
        rr_truncated = ET.SubElement(r_obj, 'truncated')
        rr_truncated.text = str(0)
        rr_difficult = ET.SubElement(r_obj, 'difficult')
        rr_difficult.text = str(ann['difficult'])

        rr_bndbox = ET.SubElement(r_obj, 'bndbox')
        rrr_xmin = ET.SubElement(rr_bndbox, 'xmin')
        rrr_ymin = ET.SubElement(rr_bndbox, 'ymin')
        rrr_xmax = ET.SubElement(rr_bndbox, 'xmax')
        rrr_ymax = ET.SubElement(rr_bndbox, 'ymax')
        rrr_xmin.text = str(ann['xmin'])
        rrr_ymin.text = str(ann['ymin'])
        rrr_xmax.text = str(ann['xmax'])
        rrr_ymax.text = str(ann['ymax'])

    format_save_xml_file(root, xml_file)



def xml2voc(xml_file, out_file, class_names, save_empty):
    """
    将HRCS-2016的.xml标签文件转成 pascal voc标签的.xml标签文件
    Args:
        xml_file (str): HRCS-2016标签文件的路径
        out_file (str): pascal voc标签文件的输出路径
        class_names (list): 标签类别名称
        save_empty (bool): 是否保存空标签
    """
    in_file = open(xml_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    w = int(root.find('Img_SizeWidth').text)
    h = int(root.find('Img_SizeHeight').text)
    c = int(root.find('Img_SizeDepth').text)
    filename = root.find('Img_FileName').text
    file_format = root.find('Img_FileFmt').text
    image_file = f"{filename}.{file_format}"
    objects = root.find('HRSC_Objects')
    annotations = []
    for obj in objects.iter('HRSC_Object'):
        difficult = '0'
        name = 'ship'
        name.replace(' ', '-')  # 将类别名称的空格间隙替换成 '-'
        # 用于打印数据集中的总共的类别名称
        if name not in class_names:
            class_names.append(name)
        box = (float(obj.find('box_xmin').text), float(obj.find('box_ymin').text),
             float(obj.find('box_xmax').text), float(obj.find('box_ymax').text))
        annotation = {
            'name': name,
            'xmin': box[0],
            'ymin': box[1],
            'xmax': box[2],
            'ymax': box[3],
            'difficult': difficult,
        }
        annotations.append(annotation)
    if len(annotations) == 0:
        LOGGER.warning(f'no lines to save, the file: {out_file} is empty')
    if len(annotations) > 0 or save_empty:
        # 将内容写入文件
        generate_voc_xml_file(out_file, image_file, (w, h, c), annotations)


def save_classes(classes_file, class_names):
    """保存classes.txt文件"""
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(class_names) + '\n')


def run(ann_dir, output_dir, save_empty=True):
    # 数据集所包含的标签类别名称
    class_names = []
    if output_dir is None:
        output_dir = osp.join(osp.dirname(ann_dir), 'Annotations_voc')
    # 创建生成label的文件夹
    os.makedirs(output_dir, exist_ok=True)
    # 后去所有的标签源文件
    ann_files = glob.glob(ann_dir + '/*.xml')
    assert len(ann_files) > 0, f"path: {ann_dir} dose not have any .xml file"
    # 标签文件转化
    pbar = tqdm(ann_files, total=len(ann_files))
    for ann_file in pbar:
        pbar.set_description(f'process {ann_file}')
        output_filename = osp.splitext(osp.basename(ann_file))[0] + '.xml'
        output_file = osp.join(output_dir, output_filename)
        xml2voc(ann_file, output_file, class_names, save_empty)
    # save classes.txt
    classes_file = osp.join(osp.dirname(output_dir), 'classes.txt')
    class_names.sort()
    save_classes(classes_file, class_names)
    # print msg
    LOGGER.info(f"dataset total class_names: {class_names}")
    LOGGER.info(f"classes.txt saved in: {classes_file}")
    LOGGER.info(f"pascal voc type label saved in: {output_dir}")


if __name__ == "__main__":
    opt = parse_opt()
    run(ann_dir=opt.ann_dir,
        output_dir=opt.output_dir,
        save_empty=not opt.noempty)

# D:/迅雷下载/HRSC2016_dataset/HRCS-2016/Annotations