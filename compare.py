# import math
from glob import glob
from os.path import join
import sys
from matplotlib import pyplot as plt
import cv2
from natsort import natsorted
import numpy as np

LOGS_PATH = '/Users/gennosuke/Downloads/logs/hpatches_test'  # Mac
# LOGS_PATH = '/home/gennosuke/logs' # Linux

all_result_npz_paths = natsorted(glob(join(
    LOGS_PATH,
    '*',
    'result.npz'
)))

# all_result
def all_result(all_result_npz_paths):
    for path in all_result_npz_paths:
        npz = np.load(path)
        path = path.replace(LOGS_PATH+'/', '').split('_')
        print(f'{path[0]}_topk:{path[2]}')
        print(npz["mscore"].sum() / len(npz["mscore"]))
        print(npz["n_matches_in"].sum() / len(npz["n_matches_in"]))
        print("-------------------------------------")

# ex1
# akazeの特徴点を学習したsuperpointの比較
# 特徴点の上位150, 200, 600個で比較
def ex1(all_result_npz_paths):
    for path in all_result_npz_paths[1:4]:
        npz = np.load(path)
        path = path.replace(LOGS_PATH+'/', '').split('_')
        print(f'{path[0]}_topk:{path[2]}')
        print(npz["mscore"].sum() / len(npz["mscore"]))
        print(npz["n_matches_in"].sum() / len(npz["n_matches_in"]))
        print("-------------------------------------")

# ex2
# 全ての画像ペアの性能比較
# 比較対象はSuperPoint(オリジナル), SuperPoint(提案手法), akaze

def ex2(all_result_npz_paths):
    path1, path2, path3 = all_result_npz_paths[0], all_result_npz_paths[1], all_result_npz_paths[4]
    npz1, npz2, npz3 = np.load(path1), np.load(path2), np.load(path3)
    int_list = [i for i in range(576) if i % 5 == 0]
    for n in int_list:
        start, end = n, n+5
        print(start, end)
        print(npz1["mscore"][start:end].sum() / len(npz1["mscore"][start:end]))
        print(npz1["n_matches_in"][start:end].sum() / len(npz1["n_matches_in"][start:end]))
        print(npz2["mscore"][start:end].sum() / len(npz2["mscore"][start:end]))
        print(npz2["n_matches_in"][start:end].sum() / len(npz2["n_matches_in"][start:end]))
        print(npz3["mscore"][start:end].sum() / len(npz3["mscore"][start:end]))
        print(npz3["n_matches_in"][start:end].sum() / len(npz3["n_matches_in"][start:end]))
        print("-------------------------------------")

# ex3
# 数字を指定して性能比較をする
# 比較対象はSuperPoint(オリジナル), SuperPoint(提案手法), akaze

def ex3(all_result_npz_paths, start, end):
    path1, path2, path3 = all_result_npz_paths[0], all_result_npz_paths[1], all_result_npz_paths[4]
    npz1, npz2, npz3 = np.load(path1), np.load(path2), np.load(path3)
    for i in range(start, end):
        print(npz1["mscore"][i])
        print(npz1["n_matches_in"][i])
        print(npz2["mscore"][i])
        print(npz2["n_matches_in"][i])
        print(npz3["mscore"][i])
        print(npz3["n_matches_in"][i])
        print("-------------------------------------")
    print("Average")
    print(npz1["mscore"][start:end].sum() / len(npz1["mscore"][start:end]))
    print(npz1["n_matches_in"][start:end].sum() / len(npz1["n_matches_in"][start:end]))
    print(npz2["mscore"][start:end].sum() / len(npz2["mscore"][start:end]))
    print(npz2["n_matches_in"][start:end].sum() / len(npz2["n_matches_in"][start:end]))
    print(npz3["mscore"][start:end].sum() / len(npz3["mscore"][start:end]))
    print(npz3["n_matches_in"][start:end].sum() / len(npz3["n_matches_in"][start:end]))

all_result_npz_paths = natsorted(glob(join(
    LOGS_PATH,
    '*',
    'result.npz'
)))
# all_result(all_result_npz_paths)
# ex1(all_result_npz_paths)
# ex2(all_result_npz_paths)
ex3(all_result_npz_paths, 20, 25) # 視点変化
# ex3(all_result_npz_paths, 210, 215) # 照明変化


def img_tile(axis_x, axis_y, img_path, filename):
    img_list = [[] for _ in range(axis_y)]
    cnt = 0

    for i in range(1, axis_x*axis_y+1):
        img = cv2.imread(img_path[i-1])
        if i & axis_x == 0:
            cnt += 1
            img_list[cnt].append(cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5))
        else:
            img_list[cnt].append(cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5))

    def concat_tile(im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
    out = concat_tile(img_list)
    cv2.imwrite(filename, out)


# 照明変化の画像
light_img_path = glob('/Users/gennosuke/Downloads/test/HPatches/i_*/*.ppm')
img_tile(3, 2, light_img_path, "light.jpg")
# 視点変化の画像
view_img_path = glob('/Users/gennosuke/Downloads/test/HPatches/v_*/*.ppm')
img_tile(3, 2, view_img_path, "view.jpg")
