import glob
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import json

# plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']#解决matplotlib无法输出中文，后面有教程
plt.rcParams['axes.unicode_minus'] = False


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def load_voc_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            if xmax == xmin or ymax == ymin:
                print(xml_file)
            dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)


def load_coco_dataset(json_path):
    dataset = []
    data = json.load(open(json_path, 'r'))

    for img in data['images']:
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = ann["bbox"]
                w = box[2]
                h = box[3]
                # dw = 1. / (img_width)
                # dh = 1. / (img_height)
                # w = w * dw
                # h = h * dh
                for i in range(10):
                    # if w<100 or h<100:
                    #     randnum1 = random.randint(30,80)
                    #     randnum2 = random.randint(30,80)
                    #     randw = random.randint(0, randnum1)
                    #     randh = random.randint(0, randnum1)
                    # else:
                    #     randw = random.randint(-99, 99)
                    #     randh = random.randint(-99, 99)
                    # dataset.append([w + randw, h + randh])
                    dataset.append([w, h])
    return np.array(dataset)


if __name__ == '__main__':
    ANNOTATIONS_PATH = r"path\to\COCO\annotations\instances_train2017.json"

    # print(__file__)
    # data = load_voc_dataset(ANNOTATIONS_PATH)
    data = load_coco_dataset(ANNOTATIONS_PATH)

    # 数据案例：
    # out = np.array([[173.03349304, 343.2177124],
    #                 [462.42819214, 423.49911499],
    #                 [277.52514648, 187.22473145],
    #                 [94.65778351, 100.66867828]])
    out = np.array()
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}-{}".format(out[:, 0], out[:, 1]))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
