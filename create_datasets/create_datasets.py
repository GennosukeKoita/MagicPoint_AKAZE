# ex1
# cocoデータセットの画像を使用して、特徴点検出を行う
# 特徴点の160*120サイズに変更した位置座標と160*120のサイズの画像を保存する
# 保存形式は画像はグレー画像でpngファイル、位置座標はnpyファイルにて保存
# 特徴点が検出されない場合はデータセットには含めないようにする

# ex2
# SuperPointに直接akazeのデータセットを読み込ませるためのデータセット作成
# npzファイルのみ存在すれば良い
# なお、SuperPointでは320×240を採用しているため最終的には320×240の場合の特徴点座標を出力するようにする
# npzファイルの中身として、平面座標(y, x)に加え強度（特徴点の強さ）を含んだ3×Nの三次元座標である

# ex3
# Unetに使用するデータセット
# 画像はCityScapesデータセットを用いた
# ↓↓↓作成方法↓↓↓
# http://cvlab.cs.miyazaki-u.ac.jp/laboratory/2021/yamamoto_honbun.pdfの7~8ページを参照


import platform
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from os import makedirs
from os.path import join, basename, splitext
from glob import glob
import sys
import natsort

# OSの選択
pf = platform.system()
if pf == "Darwin":  # Mac
    DATA_PATH = '/Users/gennosuke/Downloads/datasets'  # 環境に合わせて変える
    LOG_PATH = '/Users/gennosuke/Downloads/logs'
elif pf == "Linux":  # Linux
    DATA_PATH = '/home/gennosuke/datasets'  # 環境に合わせて変える
    LOG_PATH = '/home/gennosuke/logs'


def ex3_export_unet_dataset():
    def img_rotate(img):  # 画像の回転
        height = img.shape[0]
        width = img.shape[1]
        angle = 30  # 画像の回転する角度
        center = (int(width/2), int(height/2))
        trans = cv2.getRotationMatrix2D(center, angle, scale=1)
        image = cv2.warpAffine(img, trans, (width, height))
        return image

    # ディレクトリの作成
    dataset_path = join(DATA_PATH, "unet_dataset")
    makedirs(dataset_path, exist_ok=True)
    dirs = []
    for im in ["imgs", "masks"]:
        dirs.append(join(dataset_path, im))
        makedirs(join(dataset_path, im), exist_ok=True)

    tasks = ['train']

    for task in tasks:
        cs_path = join(DATA_PATH, "CityScapes", task, "*", "*")
        paths = glob(cs_path)
        for path in tqdm(paths):
            # 画像の読み込み
            img1 = cv2.resize(cv2.imread(path), (512, 256))
            width, height = img1.shape[1], img1.shape[0]
            img2 = img_rotate(img1)

            # akazeの実行
            akaze = cv2.AKAZE_create()
            kp1, des1 = akaze.detectAndCompute(img1, None)
            kp2, des2 = akaze.detectAndCompute(img2, None)
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # マスク画像の作成
            mask_size = (height, width)
            mask_img = np.zeros(mask_size, dtype=np.uint8)
            for match in matches:
                x, y = int(kp1[match.queryIdx].pt[0]), int(
                    kp1[match.queryIdx].pt[1])
                cv2.rectangle(
                    mask_img,
                    (x-3, y-3),
                    (x+3, y+3),
                    color=(255, 255, 255),
                    thickness=-1
                )

            filename = basename(path)
            cv2.imwrite(join(dirs[0], filename), img1)  # 元画像の保存
            cv2.imwrite(join(dirs[1], filename), mask_img)  # マスク画像の保存


def ex3_convert_superpoint_dataset():  # 途中
    # magicpoint:ディレクトリの作成
    tasks = {'train': 'training', 'val': 'validation'}
    mp_dataset_path = join(DATA_PATH, "akaze_with_unet")  # magicpoint用のデータセット
    makedirs(mp_dataset_path, exist_ok=True)
    mp_second_dataset_path = join(mp_dataset_path, "city_scapes")
    makedirs(mp_second_dataset_path, exist_ok=True)
    mp_dirs = []
    for ip in ["images", "points"]:
        mp_dirs.append(join(mp_second_dataset_path, ip))
        makedirs(join(mp_second_dataset_path, ip), exist_ok=True)

    # COCO(Superpoint):ディレクトリの作成
    coco_dataset_path = join(DATA_PATH, "unet_coco")
    makedirs(coco_dataset_path, exist_ok=True)

    for key, value in tasks.items():
        # magicpoint:ディレクトリの作成
        mp_key_path = join(mp_dirs[0], value)
        makedirs(mp_key_path, exist_ok=True)

        # COCO(Superpoint):ディレクトリの作成
        coco_key_path = join(coco_dataset_path, f'{key}2014')
        makedirs(coco_key_path, exist_ok=True)

        # 画像の読み込み、保存とunet出力から特徴点の位置座標の特定
        cs_path = join(DATA_PATH, "CityScapes", key, "*", "*")
        paths = glob(cs_path)
        for path in tqdm(paths):
            filename = basename(path)
            original_img = cv2.imread(path)
            # magicpoint:リサイズした元画像の保存
            cv2.imwrite(
                join(mp_key_path, filename),
                cv2.resize(original_img, (160, 120))
            )

            # COCO(Superpoint):元画像の保存
            cv2.imwrite(join(coco_key_path, filename), original_img)


def ex2():
    def detect_feature_points_with_resize(img):
        img = cv2.resize(img, (480, 640))  # Hは480、Wは640にしろ！
        # AKAZE検出器を読み込む
        akaze = cv2.AKAZE_create()
        # 特徴点の検出
        kp = akaze.detect(img)
        if len(kp) > 0:
            kp_list = []
            for k in kp:
                # [y,x,response]responseは特徴点の強度
                kp_list.append([k.pt[1], k.pt[0], k.response])
            kp_list = np.array(kp_list)
            kp_list = kp_list[np.argsort(kp_list[:, 2])[::-1]]  # 降順にする
            kp_list[:, 0] = kp_list[:, 0] / 2  # 480 / 2 = 320pixelに変更
            kp_list[:, 1] = kp_list[:, 1] / 2  # 640 / 2 = 240pixelに変更
            return True, kp_list
        else:
            return False, None


    tasks = ['train', 'val']
    dataset_path = join(LOG_PATH, 'akaze_coco_dataset')
    makedirs(dataset_path, exist_ok=True)
    top_k = ['all', 150, 200, 600]
    for top in top_k:
        prediction_path = join(dataset_path, f'{top}_predictions')
        makedirs(prediction_path, exist_ok=True)
        for key in tasks:
            key_path = join(prediction_path, key)
            makedirs(key_path, exist_ok=True)


    for key in tasks:
        coco_path = join(DATA_PATH, 'COCO', f'{key}2014', '*')
        coco_img_paths = natsort.natsorted(glob.glob(coco_path))
        for path in tqdm(coco_img_paths):
            basename_without_ext = splitext(basename(path))[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            judge, kp_list = detect_feature_points_with_resize(img)
            for top in top_k:
                if judge:
                    if top == 'all':
                        np.savez(
                            f'{dataset_path}/{top}_predictions/{key}/{basename_without_ext}.npz', pts=kp_list)
                    else:
                        np.savez(
                            f'{dataset_path}/{top}_predictions/{key}/{basename_without_ext}.npz', pts=kp_list[:top])


def ex1():
    def detect_feature_points_with_resize(img, a):
        height, width, _ = img.shape
        img = cv2.resize(img, (width*a, height*a))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        akaze = cv2.AKAZE_create()
        kp = akaze.detect(img)
        if len(kp) > 0:
            kp_list = []
            for k in kp:
                kp_list.append([k.pt[1], k.pt[0], k.response]) # [y,x,response]responseは特徴点の強度
            kp_list = np.array(kp_list)
            # height*aサイズの位置座標 / (height*a) * 120 = 120pixelに変更
            kp_list[:, 0] = kp_list[:, 0] * 120 / (height*a)
            # width*aサイズの位置座標 / (width*a) * 160 = 160pixelに変更
            kp_list[:, 1] = kp_list[:, 1] * 160 / (width*a)
            kp_list = np.round(kp_list, decimals=3)
            resize_img = cv2.resize(img, (160, 120))
            return True, resize_img, kp_list[:,:2]
        else:
            return False, None, None

    # ディレクトリの作成
    dataset_path = f'{DATA_PATH}/akaze_coco'
    makedirs(dataset_path, exist_ok=True)
    base_path = join(dataset_path, 'coco')
    makedirs(base_path, exist_ok=True)
    for ip in ["images", "points"]:
        ip_path = join(base_path, ip)
        makedirs(ip_path, exist_ok=True)
        for data in ["training", "validation"]:
            data_path = join(ip_path, data)
            makedirs(data_path, exist_ok=True)

    # データセットの作成と保存
    tasks = {'training':"train2014", 'validation':"val2014"}
    for key, value in tasks.items():
        coco_path = join(DATA_PATH, 'COCO', value, '*')
        save_img_path = join(base_path, 'images', key)
        save_pnt_path = join(base_path, 'points', key)
        coco_img_paths = natsort.natsorted(glob(coco_path))
        for path in tqdm(coco_img_paths):
            basename_without_ext = splitext(basename(path))[0]
            img = cv2.imread(path)
            judge, resize_img, kp_list = detect_feature_points_with_resize(img, 2) # [y,x,stregth]
            if judge:
                cv2.imwrite(f'{save_img_path}/{basename_without_ext}.png', resize_img)
                np.save(f'{save_pnt_path}/{basename_without_ext}.npy', kp_list)


def main():
    args = sys.argv
    if args[1] == "unet":
        ex3_export_unet_dataset()
    elif args[1] == "super":
        ex3_convert_superpoint_dataset()
    elif args[1] == "ex2":
        ex2()
    elif args[1] == "ex1":
        ex1()


main()
