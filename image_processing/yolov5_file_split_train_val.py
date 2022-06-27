"""
根据原数据集中train和val的数据，将数据集分成训练集和验证机
all data -> train data & val data
using original val.csv

数据集架构
dataset
    - images
        -train
        -val
    - lables
        -train
        -val
"""
import os
import cv2
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np


def split_data(csv_path,dataset_path):
    csv_data = pd.read_csv(csv_path)
    for index, row in tqdm(csv_data.iterrows()):
        img_name = row['patient_id'] + '_' + row['left or right breast'][0] + '_' + row['image view'] + '.png'
        img_path = os.path.join(dataset_path,'images','train',img_name)
        if os.path.exists(img_path):
            img_save_path = os.path.join(dataset_path,'images','val')
            lab_read_path = os.path.join(dataset_path,'lables','train',img_name.split('.')[0]+'.txt')
            lab_save_path = os.path.join(dataset_path,'lables','val')
            shutil.move(img_path,img_save_path)
            shutil.move(lab_read_path,lab_save_path)


if __name__ == "__main__":
    calc_csv_path = ''
    mass_csv_path = ''
    dataset_path = ''
    split_data(calc_csv_path,dataset_path)
    split_data(mass_csv_path,dataset_path)


