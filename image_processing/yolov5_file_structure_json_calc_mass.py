import os
import json
import cv2
import pandas as pd
import numpy as np

def add_prefix(path):
    return os.path.join('/home/extend/datasets', path)

def create_txt(json_path, image_path, text_path, save_img_path):
    """
    给图片创建符合yolov5的标签，同时将图片和标签移动到对应目录
    :param json_path:
    :param image_path:
    :param text_path:
    :param save_img_path:
    :return:
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    img_name = data['filename']
    # image_path = os.path.join(image_path, img_name + '.png')
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    img_shape = image.shape  # image.shape -> (h,w)
    img_save_path = os.path.join(save_img_path, image_path.split('/')[-1])
    cv2.imwrite(img_save_path,image)  # 图片保存在对应位置
    roi_list = list(data['ROI'])
    text_file = img_name + '.txt'
    with open(os.path.join(text_path, text_file), 'w') as file:
        lines = ''
        for item in roi_list:
            box = item['bbox']  # [[x,y][w,h]]
            center_x = box[0][0] / img_shape[1]
            center_y = box[0][1] / img_shape[0]
            width = box[1][0] / img_shape[1]
            height = box[1][1] / img_shape[0]

            if item['lesion_type'] == "mass":
                gt_classes = '0'
            else:
                gt_classes = '1'

            pathology = '1' if item['pathology'] == 'MALIGNANT' else '0'
        lines += gt_classes + ' ' + center_x + ' ' + center_y + ' ' + width + ' ' + height + ' ' + pathology + '\n'
        file.write(lines)
    print(data)


def iterate_file(path):
    text_path = '/home/extend2/datasets/ly_cbis_ddsm_yolov5_json_original_calc_mass/labels/train'
    save_img_path = '/home/extend2/datasets/ly_cbis_ddsm_yolov5_json_original_calc_mass/images/train'
    if not os.path.exists(path):
        print("File not exist")
        return False
    dir_list = os.listdir(path)
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        if os.path.isdir(dir_path):
            file_list = os.listdir(dir_path)
            for file in file_list:
                if file.split('.')[-1] == 'json':
                    file_path = os.path.join(dir_path, file)
                    img_path = file_path.split('.')[0]+'.png'
                    create_txt(file_path, img_path, text_path, save_img_path)


if __name__ == "__main__":
    # json_path = r"./P_00001_L_CC.json"
    # image_path = r"D:\SourcetreeSpace\Python_function\image_processing"
    # text_path = r'D:\SourcetreeSpace\Python_function\image_processing'
    # create_txt(json_path, image_path, text_path)
    iterate_file()