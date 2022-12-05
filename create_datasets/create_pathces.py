from os.path import basename, join, splitext
from os import makedirs
import cv2
from natsort import natsorted
from tqdm import tqdm
from glob import glob
import numpy as np
from itertools import product
import sys

DATA_PATH = '/home/gennosuke/datasets'

city_path = glob(join(DATA_PATH, 'city_pathces', '*'))
half = int(len(city_path)/2)
p_path = join(DATA_PATH, 'HPatches_city')
makedirs(p_path, exist_ok=True)

# v_patterns = list(product(
#     [10, 30, 45, 90], # 角度の変化
#     [0.5, 0.8, 1.0, 1.2, 1.5] # スケールの変化
#     ))
# for i, path in enumerate(city_path[:half]):
#     img = cv2.imread(path)
#     height, width, _ = img.shape
#     for j, p in enumerate(v_patterns):
#         save_path = join(p_path, f'v_{i}')
#         makedirs(save_path, exist_ok=True)

#         #画像の回転行列
#         rot_matrix = cv2.getRotationMatrix2D((width/2,height/2),  # 回転の中心座標
#                                             p[0],                 # 回転する角度
#                                             p[1],                 # 画像のスケール
#                                             )

#         # アファイン変換適用
#         afin_img = cv2.warpAffine(img,             # 入力画像
#                                 rot_matrix,        # 行列
#                                 (width,height)     # 解像度
#                                 )
        
#         cv2.imwrite(f'{save_path}/{j+2}.ppm', afin_img)
#         f = open(join(save_path, f'H_1_{j+2}'), 'w')
#         f.write('1 0 0\n0 1 0\n0 0 1')
#         f.close()
#     cv2.imwrite(f'{save_path}/1.ppm', img)



# **********************************************************
# 【ガンマ補正の公式】
#   Y = 255(X/255)**(1/γ)

# 【γの設定方法】
#   ・γ>1の場合：画像が明るくなる
#   ・γ<1の場合：画像が暗くなる
# **********************************************************


i_patterns = [0.5, 0.8, 1.2, 1.5, 3.0]
# 照明の変化
for i, path in enumerate(city_path[half:]):
    img = cv2.imread(path)
    height, width, _ = img.shape
    for j, p in enumerate(i_patterns):
        save_path = join(p_path, f'i_{i}')
        makedirs(save_path, exist_ok=True)
        
        img2gamma = np.zeros((256,1),dtype=np.uint8)  # ガンマ変換初期値
        # 公式適用
        for k in range(256):
            img2gamma[k][0] = 255 * (float(k)/255) ** (1.0 /p)

        # 読込画像をガンマ変換
        gamma_img = cv2.LUT(img,img2gamma)

        cv2.imwrite(f'{save_path}/{j+2}.ppm', gamma_img)
        f = open(join(save_path, f'H_1_{j+2}'), 'w')
        f.write('1 0 0\n0 1 0\n0 0 1')
        f.close()
    cv2.imwrite(f'{save_path}/1.ppm', img)