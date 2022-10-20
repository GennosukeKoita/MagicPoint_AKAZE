# SuperPointに直接akazeのデータセットを読み込ませるためのデータセット作成
# npzファイルのみ存在すれば良い
# なお、SuperPointでは320×240を採用しているため最終的には320×240の場合の特徴点座標を出力するようにする
# npzファイルの中身として、平面座標(y, x)に加え強度（特徴点の強さ）を含んだ3×Nの三次元座標である

import numpy as np
import cv2 as cv
import os
from os.path import join, basename, splitext
import glob
import natsort
import matplotlib.pyplot as plt
import platform
from tqdm import tqdm

pf = platform.system()
if pf == 'Darwin': # Mac
    DATA_PATH = '../datasets'
    LOG_PATH = '../logs'
elif pf == 'Linux': # Linux
    DATA_PATH = '/home/gennosuke/datasets'
    LOG_PATH = '/home/gennosuke/logs'

def detect_feature_points_with_resize(img):
    img = cv.resize(img, (480, 640)) # Hは480、Wは640にしろ！
    # AKAZE検出器を読み込む
    akaze = cv.AKAZE_create()
    # 特徴点の検出
    kp = akaze.detect(img)
    if len(kp) > 0:
        kp_list = []
        for k in kp:
            kp_list.append([k.pt[1], k.pt[0], k.response]) # [y,x,response]responseは特徴点の強度
        kp_list = np.array(kp_list)
        kp_list = kp_list[np.argsort(kp_list[:,2])[::-1]] # 降順にする
        kp_list[:,0] = kp_list[:,0] / 2 # 480 / 2 = 320pixelに変更
        kp_list[:,1] = kp_list[:,1] / 2 # 640 / 2 = 240pixelに変更
        return True, kp_list
    else:
        return False, None

# akaze_coco_dataset/prediction/train/*.npz
# akaze_coco_dataset/prediction/val/*.npz
tasks = ['train', 'val']
dataset_path = f'{LOG_PATH}/akaze_coco_dataset'
os.makedirs(dataset_path, exist_ok=True)
top_k = ['all', 150, 200, 600]
for top in top_k:
    prediction_path = join(dataset_path, f'{top}_predictions')
    os.makedirs(prediction_path, exist_ok=True)
    for key in tasks:
        key_path = join(prediction_path, key)
        os.makedirs(key_path, exist_ok=True)


for key in tasks:
    coco_path = f'{DATA_PATH}/COCO/{key}2014/*'
    coco_img_paths = natsort.natsorted(glob.glob(coco_path))
    for path in tqdm(coco_img_paths):
        basename_without_ext = splitext(basename(path))[0]
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        judge, kp_list = detect_feature_points_with_resize(img)
        for top in top_k:
            if judge:
                if top == 'all':
                    np.savez(f'{dataset_path}/{top}_predictions/{key}/{basename_without_ext}.npz', pts=kp_list)
                else:
                    np.savez(f'{dataset_path}/{top}_predictions/{key}/{basename_without_ext}.npz', pts=kp_list[:top])