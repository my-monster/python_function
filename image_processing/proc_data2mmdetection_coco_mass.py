import os
import json
import cv2
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

img_id = 0
ann_id = 0

line_color = [(0, 255, 0), (255, 0, 0), (255, 0, 255), (255, 255, 0),
              (255, 255, 255), (0, 0, 255), (205, 92, 92), (238, 121, 66),
              (153, 50, 204), (238, 58, 140), (202, 255, 112), (65, 105, 225)]

func = lambda x: [y for l in x for y in func(l)] if type(x) is list else [x]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def collect_image_annotation(read_json_path,read_img_path,save_img_path):
    global img_id # 使用global声明，在函数中就可以修改全局变量的值
    global ann_id
    with open(read_json_path,'r') as read_json:
        read_json_data = json.load(read_json)
    image = {}
    annotation = []
    # img_path = str(read_img_path.joinpath('.',read_json_data['filename']+'.png'))
    # img_path = os.path.join(read_img_path,read_json_data['filename']+'.png')
    img = cv2.imread(read_img_path, cv2.IMREAD_ANYDEPTH)
    img_shape = img.shape
    img_look = image.copy()
    # shutil.copy(img_path,save_img_path)
    image = {
        'file_name': read_json_data['filename']+'.png',
        'height': img_shape[0],
        'width': img_shape[1],
        'id': img_id
    }

    ann_len = len(read_json_data['ROI'])
    for idx in range(ann_len):
        roi = read_json_data['ROI'][idx]
        if roi['lesion_type'] != 'mass':
            continue
        segmentation = [func(roi['points'])]
        # area = ann['area']
        bbox = roi['bbox']
        x_min = bbox[0][0] - bbox[1][0] / 2
        y_min = bbox[0][1] - bbox[1][1] / 2
        width = bbox[1][0]
        height = bbox[1][1]
        x_max = x_min + width
        y_max = y_min + height

        # 标签可视化
        c1, c2 = (int(x_min), int(y_min)), (int(x_max), int(y_max))
        color = line_color[0]
        cv2.rectangle(img_look, c1, c2, color, thickness=2)
        # label
        tl = 1
        tf = 2  # font thickness
        i = 4

        if roi['lesion_type'] != 'mass':
            text = 'mass' + ' ' + roi['pathology']
        else:
            text = 'calc' + ' ' + roi['pathology']

        t_size = cv2.getTextSize(text, 0, fontScale=tl, thickness=tf)[0]
        cv2.putText(img_look, text, (c1[0], c1[1] - 2 - (i - 4) * t_size[1]), 0, tl, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)

        area = width * height
        ann = {
            'segmentation': segmentation,
            'area': area,
            'iscrowd': 0,
            'image_id': img_id,
            'bbox': [x_min,y_min,width,height],
            'category_id': 0 if roi['lesion_type'] == 'mass' else 1,
            'id': ann_id
        }
        annotation.append(ann)
        # print(ann)
        ann_id = ann_id + 1
    img_id = img_id + 1

    output_image_patch_look_path = os.path.join(save_img_path+'_look', read_json_data['filename']+'.png')
    # print(annotation)
    # print(x_min)
    # print(ann_id)
    # print(read_json_data)
    # 判断当前图片中是否有mass病灶，没有则返回 空
    if not annotation:
        return {}, []
    else:
        cv2.imwrite(output_image_patch_look_path, img_look)
        shutil.copy(read_img_path, save_img_path)
        return image, annotation


def init_save_json(save_json_path):
    with open(save_json_path, 'w') as save_json:
        dict1 = {}
        dict1['images'] = []
        dict1['annotations'] = []
        dict1['categories'] = []
        json.dump(dict1, save_json, cls=NpEncoder)
        print('Init succeed!')


def data2coco(save_train_json_path,save_val_json_path,path,save_train_img_path,save_val_img_path,judge_path):
    # with open(save_json_path, 'r') as save_json:
    #     save_json_data = json.load(save_json)
    proc_img_path = ''
    train_image = []
    train_annotation = []
    val_image = []
    val_annotation = []
    if not os.path.exists(path):
        print("File not exist")
        return False
    dir_list = os.listdir(path)
    for dir in tqdm(dir_list):
        dir_path = os.path.join(path, dir)
        if os.path.isdir(dir_path):
            file_list = os.listdir(dir_path)
            for file in file_list:
                if file.split('.')[-1] == 'json':
                    file_path = os.path.join(dir_path, file)
                    # img_path = file_path.split('.')[0] + '.png'
                    # img_path = os.path.join(proc_img_path, file.split('.')[0]+'.png')
                    judge_yolo_path = os.path.join(judge_path,file.split('.')[0]+'.png')
                    exists = os.path.exists(judge_yolo_path)
                    if exists:
                        img_path = os.path.join(proc_img_path, 'train_image_proc', file.split('.')[0] + '.png')
                        image, annotation = collect_image_annotation(file_path,img_path,save_train_img_path)
                        if annotation:
                            train_image.append(image)
                            train_annotation.extend(annotation)
                    else:
                        img_path = os.path.join(proc_img_path, 'val_image_proc', file.split('.')[0] + '.png')
                        image, annotation = collect_image_annotation(file_path, img_path, save_val_img_path)
                        if annotation:
                            val_image.append(image)
                            val_annotation.extend(annotation)
    with open(save_train_json_path, 'r') as save_json:
        save_json_data = json.load(save_json)
        save_json_data['images'] = train_image
        save_json_data['annotations'] = train_annotation
        with open(save_train_json_path, 'w') as f:
            json.dump(save_json_data, f, cls=NpEncoder)
    with open(save_val_json_path, 'r') as save_json:
        save_json_data = json.load(save_json)
        save_json_data['images'] = val_image
        save_json_data['annotations'] = val_annotation
        with open(save_val_json_path, 'w') as f:
            json.dump(save_json_data, f, cls=NpEncoder)


def save_categories(json_path):
    with open(json_path,'r') as save_json:
        json_data = json.load(save_json)
        json_data['categories'] = [
            {
                'id': 0,
                'name': 'mass'
            # },
            # {
            #     'id': 1,
            #     'name': 'calc'
            }
        ]
        with open(json_path,'w') as f:
            json.dump(json_data, f, cls=NpEncoder)


if __name__ == '__main__':
    # read_json_path = Path('./P_00001_L_CC.json')
    # read_img_path = Path('D:\SourcetreeSpace\Python_function\image_processing')
    # save_img_path = Path('./')
    # collect(read_json_path,read_img_path,save_img_path)
    save_train_json_path = '/home/extend/datasets/cbis-ddsm_mmdetection_coco_mass/annotations/train_coco.json'
    save_val_json_path = '/home/extend/datasets/cbis-ddsm_mmdetection_coco_mass/annotations/val_coco.json'
    path = '/home/extend/datasets/cbis-ddsm_all/all'
    save_train_img_path = '/home/extend/datasets/cbis-ddsm_mmdetection_coco_mass/train_image'
    save_val_img_path = '/home/extend/datasets/cbis-ddsm_mmdetection_coco_mass/val_image'
    judge_path = '/home/extend/datasets/cbis-ddsm_mmdetection_coco_mass_calc/train_image'
    # init_save_json(save_train_json_path)
    # init_save_json(save_val_json_path)
    # data2coco(save_train_json_path,save_val_json_path,path,save_train_img_path,save_val_img_path,judge_path)
    save_categories(save_train_json_path)
    save_categories(save_val_json_path)