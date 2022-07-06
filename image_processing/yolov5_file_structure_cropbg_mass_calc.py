"""
基于yolov5_file_structure，使用cbis-ddsm，裁剪空白区域，生成新的bounding-box
"""

import os
import pandas as pd
import cv2
import numpy as np

line_color = [(0, 255, 0), (255, 0, 0), (255, 0, 255), (255, 255, 0),
              (255, 255, 255), (0, 0, 255), (205, 92, 92), (238, 121, 66),
              (153, 50, 204), (238, 58, 140), (202, 255, 112), (65, 105, 225)]

def add_prefix(path):
    return os.path.join('/home/extend/datasets', path)

def apply_CLAHE(img): # CLAHE
    # clipLimit截断的值，对于频率超过了阈值的灰度级，那么就把这些超过阈值的部分裁剪掉平均分配到各个灰度级上
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(32, 32))
    cl1 = clahe.apply(img)
    cl1 = cl1 - cl1.min()
    # res = np.hstack((img, cl1))
    # cv2.imwrite('res.png', res.astype('uint16'))
    return cl1

def create_txt(csv_path, text_path, save_img_path, save_look_path, enhance=False):
    df = pd.read_csv(csv_path)
    df['image_path'] = df['image_path'].apply(add_prefix)
    for index, df_item in df.iterrows():
        file_path = df_item.image_path
        file_path = file_path.replace('image', 'image_delbg')

        print(file_path)
        file_save_path = os.path.join(save_img_path, file_path.split('/')[-1])
        image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

        rows_sum = np.sum(image, axis=0)
        cols_sum = np.sum(image, axis=1)
        x_le_top_idx = np.where(rows_sum != 0)[0][0]
        x_ri_down_idx = np.where(rows_sum != 0)[0][-1]
        y_le_top_idx = np.where(cols_sum != 0)[0][0]
        y_ri_down_idx = np.where(cols_sum != 0)[0][-1]

        crop_image = image[y_le_top_idx:y_ri_down_idx, x_le_top_idx:x_ri_down_idx]
        if enhance:
            crop_image = apply_CLAHE(crop_image)
        cv2.imwrite(file_save_path, crop_image)
        img_look = crop_image.copy()

        location = list(eval(df_item.location))
        img_shape = crop_image.shape
        text_file = df_item.image_id + '.txt'
        with open(os.path.join(text_path, text_file), 'w') as file:
            lines= ''
            for item in location:
                box = item['bbox'] # 左上右下，并且高为x
                pathology = '1' if item['pathology'] == 'MALIGNANT' else '0'
                box[1] = box[1] - x_le_top_idx
                box[3] = box[3] - x_le_top_idx
                box[0] = box[0] - y_le_top_idx
                box[2] = box[2] - y_le_top_idx

                # cv2.rectangle(img_look, (box[1], box[0]), (box[3], box[2]), 255, thickness=2)
                c1, c2 = (int(box[1]), int(box[0])), (int(box[3]), int(box[2]))
                color = line_color[0]
                cv2.rectangle(img_look, c1, c2, color, thickness=2)
                # label
                tl = 1
                tf = 2  # font thickness
                i = 4

                if 'Mass' in csv_path:
                    text = 'mass' + ' ' + item['pathology']
                else:
                    text = 'calc' + ' ' + item['pathology']

                t_size = cv2.getTextSize(text, 0, fontScale=tl, thickness=tf)[0]
                cv2.putText(img_look, text, (c1[0], c1[1] - 2 - (i - 4) * t_size[1]), 0, tl, [225, 255, 255],
                            thickness=tf,
                            lineType=cv2.LINE_AA)  # (c1[0], c1[1] - 2) is Bottom-left corner of the text string in the image

                box[1] = box[1] / img_shape[1]
                box[3] = box[3] / img_shape[1]
                box[0] = box[0] / img_shape[0]
                box[2] = box[2] / img_shape[0]

                if 'Mass' in csv_path:
                    gt_classes = '0'
                else:
                    gt_classes = '1'
                # gt_classes = '0'
                center_x = '{:.6f}'.format((box[1] + box[3]) / 2)
                center_y = '{:.6f}'.format((box[0] + box[2]) / 2)
                width = '{:.6f}'.format(box[3] - box[1])
                height = '{:.6f}'.format(box[2] - box[0])
                lines += gt_classes + ' ' + center_x + ' ' + center_y + ' ' + width + ' ' + height  + ' ' + pathology + '\n'
            file.write(lines)
        output_image_patch_look_path = os.path.join(save_look_path, file_path.split('/')[-1])
        cv2.imwrite(output_image_patch_look_path, img_look)
    pass


if __name__ == '__main__':
    train_csv_file = '/home/extend/datasets/cbis_ddsm_original_png/Mass-training/cbis_ddsm.csv'
    val_csv_file = '/home/extend/datasets/cbis_ddsm_original_png/Mass-test/cbis_ddsm.csv'
    train_save_text = '/home/extend/datasets/cbis_ddsm_yolov5_mass_calc_add_crop/labels/train'
    val_save_text = '/home/extend/datasets/cbis_ddsm_yolov5_mass_calc_add_crop/labels/val'
    train_save_img = '/home/extend/datasets/cbis_ddsm_yolov5_mass_calc_add_crop/images/train'
    val_save_img = '/home/extend/datasets/cbis_ddsm_yolov5_mass_calc_add_crop/images/val'
    train_save_look = '/home/extend/datasets/cbis_ddsm_yolov5_mass_calc_add_crop/look/train'
    val_save_look = '/home/extend/datasets/cbis_ddsm_yolov5_mass_calc_add_crop/look/val'
    os.makedirs(train_save_text, exist_ok=True)
    os.makedirs(val_save_text, exist_ok=True)
    os.makedirs(train_save_img, exist_ok=True)
    os.makedirs(val_save_img, exist_ok=True)
    os.makedirs(train_save_look, exist_ok=True)
    os.makedirs(val_save_look, exist_ok=True)

    # 肿块，label0
    create_txt(val_csv_file, val_save_text, val_save_img, val_save_look)
    create_txt(train_csv_file, train_save_text, train_save_img, train_save_look)

    # 钙化，label1
    train_csv_file = '/home/extend/datasets/cbis_ddsm_original_png/Calc-training/cbis_ddsm.csv'
    val_csv_file = '/home/extend/datasets/cbis_ddsm_original_png/Calc-test/cbis_ddsm.csv'
    create_txt(val_csv_file, val_save_text, val_save_img, val_save_look)
    create_txt(train_csv_file, train_save_text, train_save_img, train_save_look)
