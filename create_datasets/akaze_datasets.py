import numpy as np
import cv2 as cv
import os
from os.path import splitext, basename, join
import glob
import natsort
from tqdm import tqdm
import matplotlib.pyplot as plt

# DATA_PATH = '/Users/gennosuke/Downloads/datasets'  # 環境に合わせて変更する
DATA_PATH = '/home/gennosuke/datasets'

# 画像サイズは全て同じにする
# グレー画像を適応する
# 特徴点が検出されない場合はデータセットには含めない
# この３つは絶対に守れ！！！！！！！！！

def resize_img_and_kp(kp):
    kp_list = []
    for k in kp:
        kp_list.append([k.pt[1], k.pt[0], k.response]) # [y,x,response]responseは特徴点の強度
    kp_list = np.array(kp_list)
    kp_list = kp_list[np.argsort(kp_list[:, 2])[::-1]]
    kp_list = np.round(kp_list, decimals=3)
    return kp_list[:,:2]

def detect_feature_points_with_resize(img, top_k=0):
    img = cv.resize(img, (320, 240))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    akaze = cv.AKAZE_create()
    kp = akaze.detect(img)
    if len(kp) > 0:
        kp_list = resize_img_and_kp(kp)
        if top_k > 0 and len(kp_list) >= top_k:
            kp_list = kp_list[:top_k]
        return True, img, kp_list
    else:
        return False, None, None

create_dataset_path = f'{DATA_PATH}/akaze_coco'
os.makedirs(create_dataset_path, exist_ok=True)
base_path = join(create_dataset_path, 'ms-coco')
os.makedirs(base_path, exist_ok=True)
for ip in ["images", "points"]:
    ip_path = join(base_path, ip)
    os.makedirs(ip_path, exist_ok=True)
    for data in ["training", "validation"]:
        data_path = join(ip_path, data)
        os.makedirs(data_path, exist_ok=True)

tasks = {'training':"train2014", 'validation':"val2014"}
for key, value in tasks.items():
    coco_path = join(DATA_PATH, 'COCO', value, '*')
    save_img_path = join(base_path, 'images', key)
    save_pnt_path = join(base_path, 'points', key)
    coco_img_paths = natsort.natsorted(glob.glob(coco_path))
    for path in tqdm(coco_img_paths):
        basename_without_ext = splitext(basename(path))[0]
        img = cv.imread(path)
        judge, resize_img, kp_list = detect_feature_points_with_resize(img, 0) # [y,x,stregth]
        if judge:
            cv.imwrite(f'{save_img_path}/{basename_without_ext}.png', resize_img)
            np.save(f'{save_pnt_path}/{basename_without_ext}.npy', kp_list)