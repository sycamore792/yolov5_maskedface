# 先保存一个上述图片路径的txt文件
import os

import cv2
import numpy as np
from PIL._imaging import font

with open('img.txt', 'w') as f:
    for dir in os.listdir(img_path):
        pic_name = img_path + dir + '\n'
        f.write(pic_name)


# 这个函数是用来获取xml中的预测框位置信息，因为txt文件已经进行了归一化，还原较麻烦
def _read_anno(filename):
    import xml.etree.ElementTree as ET

    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        bbox = obj.find('bndbox')
        x1, y1, x2, y2 = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]

        result = [x1, y1, x2, y2]
        objects.append(result)
    return objects


with open('img.txt', 'r') as f:
    path = f.readlines()

for im in path:
    im = im.rstrip('\n')
    filename = os.path.basename(im)
    name = filename.split('.')[0]
    # 这里第一个replace根据自己的数据集位置进行修改
    txt_path = im.replace().replace('jpg', 'txt').rstrip('\n')
    xml_path = im.replace().replace('jpg', 'xml')
    box = _read_anno(xml_path)
    box = np.array(box)
    labels = []
    # 获取label
    with open(txt_path, 'r') as f:
        for line in f:
            label = line.split(' ')[0]
            labels.append(label)

    img = cv2.imread(im)
    for i in range(0, len(labels)):
        # 画图并标上label
        cv2.rectangle(img, (box[i, 0], box[i, 1]), (box[i, 2], box[i, 3]), (0, 0, 255), thickness=2)
        cv2.putText(img, '{}'.format(labels[i]), (box[i, 0], box[i, 1]), font, 0.8, (0, 0, 0), 2)
    cv2.imwrite('{}.jpg'.format(name), img)
