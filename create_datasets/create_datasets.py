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
from random import sample
import pprint
import matplotlib.pyplot as plt


def img_rotate(img, angle):  # 画像の回転
    height = img.shape[0]
    width = img.shape[1]
    center = (int(width/2), int(height/2))
    trans = cv2.getRotationMatrix2D(center, angle, scale=1)
    image = cv2.warpAffine(img, trans, (width, height))
    return image, center

def rotation(kp, r_axis, t, deg=True):
    # 度数単位の角度をラジアンに変換
    if deg == True:
        t = np.deg2rad(t)
    r_axis = np.array(r_axis)
    # 回転行列
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t),  np.cos(t)]])
    return np.dot(R, kp-r_axis)+r_axis

# OSの選択
pf = platform.system()
if pf == "Darwin":  # Mac
    DATA_PATH = '/Users/gennosuke/Downloads/datasets'  # 環境に合わせて変える
    LOG_PATH = '/Users/gennosuke/Downloads/logs'
elif pf == "Linux":  # Linux
    DATA_PATH = '/home/gennosuke/datasets'  # 環境に合わせて変える
    LOG_PATH = '/home/gennosuke/logs'


def ex3_sp_dataset():
    tasks = {'train': ['training', 44800], 'val': ['validation', 6400]}
    # COCO(Superpoint):ディレクトリの構築
    coco_dataset_path = join(DATA_PATH, "ex3_coco")
    makedirs(coco_dataset_path, exist_ok=True)
    for key, value in tasks.items():
        # COCO(Superpoint):ディレクトリの構築
        coco_key_path = join(coco_dataset_path, f'{key}2014')
        makedirs(coco_key_path, exist_ok=True)
        img_paths = glob(join(DATA_PATH, 'CityScapes', key, '*', '*'))
        for path in tqdm(img_paths):
            # COCO(Superpoint):元画像の保存
            o_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            filename = basename(path)
            cv2.imwrite(join(coco_key_path, filename), o_img)

def ex3_coco(dataset:str, angle: int, mag=None, resize_height=None, resize_width=None):
    tasks = {'train': ['training', 44800], 'val': ['validation', 6400]}
    NORM_THESHOLD = 1

    print("ディレクトリの構成中")
    if dataset == "coco":
        mpd_path = join(
            DATA_PATH,
            f'ex3_magicpoint_dataset_mag:{mag}_angle:{angle}'
        )  # magicpoint用のデータセット
    elif dataset == "city":
        # magicpoint:ディレクトリの構築
        mpd_path = join(
            DATA_PATH, 
            f'ex3_magicpoint_dataset_{angle}_height{resize_height}_width_{resize_width}'
        )  # magicpoint用のデータセット
    makedirs(mpd_path, exist_ok=True)
    mpd_path2 = join(mpd_path, "city_scapes")
    makedirs(mpd_path2, exist_ok=True)
    mp_dirs = []
    for ip in ["images", "points"]:
        mp_dirs.append(join(mpd_path2, ip))
        makedirs(join(mpd_path2, ip), exist_ok=True)
    
    # COCO(Superpoint):ディレクトリの構築
    # coco_dataset_path = join(DATA_PATH, "ex3_coco")
    # makedirs(coco_dataset_path, exist_ok=True)

    for key, value in tasks.items():
        # magicpoint:ディレクトリの構築
        mp_key_path = join(mp_dirs[0], value[0])
        makedirs(mp_key_path, exist_ok=True)
        mp_value_path = join(mp_dirs[1], value[0])
        makedirs(mp_value_path, exist_ok=True)

        # COCO(Superpoint):ディレクトリの構築
        # coco_key_path = join(coco_dataset_path, f'{key}2014')
        # makedirs(coco_key_path, exist_ok=True)
        print(f'{key}:ディレクトリの構成完了')

        if dataset == "coco":
            img_paths = natsort.natsorted(
                glob(join(DATA_PATH, 'COCO', f'{key}2014', '*')))[:value[1]]
        elif dataset == "city":
            img_paths = glob(join(DATA_PATH, 'CityScapes', key, '*', '*'))
        print(f'{key}:magicpoint, superpoint用の画像')
        for path in tqdm(img_paths):
            # 画像の読み込み
            o_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            mp_img = cv2.resize(o_img, (160, 120)) # magicpoint:画像の読み込みとリサイズ
            sp_img = o_img # superpoint:画像の読み込み
            if dataset == "coco":
                resize_height = o_img.shape[0] * mag
                resize_width = o_img.shape[1] * mag
            img1 = cv2.resize(o_img, (resize_width, resize_height))
            img2, center = img_rotate(img1, angle)

            # akazeの実行
            akaze = cv2.AKAZE_create()
            kp1, des1 = akaze.detectAndCompute(img1, None)
            kp2, des2 = akaze.detectAndCompute(img2, None)
            bf = cv2.BFMatcher(NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # マッチングした特徴点同士の二点間距離を計算する、
            # 条件として、
            # ①アフィン変換していない画像の特徴点は何にもしない
            # ②アフィン変換した画像の特徴点は画像中心周りに時計回りにθ度回転移動させた特徴点を使用
            # ③①、②の特徴点の二点間距離を計算する
            pnts = []
            for m in matches:
                kp_1 = kp1[m.queryIdx].pt
                kp_2 = rotation(kp2[m.trainIdx].pt, center, angle)
                distance = np.linalg.norm(kp_1-kp_2)
                if distance <= NORM_THESHOLD:
                    pnts.append([kp_1[1], kp_1[0]])

            if len(pnts) != 0: 
                pnts = np.array(pnts)
                pnts[:, 0] = pnts[:, 0] * 120 / resize_height
                pnts[:, 1] = pnts[:, 1] * 160 / resize_width
                pnts = np.round(pnts, decimals=3)

                filename = basename(path)
                # magicpoint:リサイズした元画像の保存と選択特徴点の位置座標の保存
                cv2.imwrite(join(mp_key_path, filename), mp_img)
                np.save(join(mp_value_path, f'{splitext(filename)[0]}.npy'), pnts)

            # COCO(Superpoint):元画像の保存
            # cv2.imwrite(join(coco_key_path, filename), sp_img)
            
        print(f'{key}:magicpoint, superpoint用の画像の保存完了')

def ex3_city(resize_height: int, resize_width: int):

    def select_imgs(tasks):
        img_paths = glob(join(DATA_PATH, 'CityScapes', '*', '*', '*'))
        for key, value in tasks.items():
            paths = sample(img_paths, value[1])
            value.append(paths)
            img_paths = [i for i in img_paths if i not in paths]
        return tasks
    
    def save_pathces_imgs(array):
        save_path = join(DATA_PATH, 'city_pathces')
        makedirs(save_path, exist_ok=True)
        for path in array[2]:
            img = cv2.imread(path)
            filename = basename(path)
            cv2.imwrite(join(save_path, filename), img)


    tasks = {'train': ['training', 4400], 'val': ['validation', 500], 'test': ['test', 100]}
    angles = [30, 45, 60, 90]
    NORM_THESHOLD = 10
    tasks = select_imgs(tasks)
    save_pathces_imgs(tasks['test'])
    for angle in angles:
        print("ディレクトリの構成中")
        # magicpoint:ディレクトリの構築
        mpd_path1 = join(DATA_PATH, f'ex3_mp_city_{angle}_height{resize_height}_width_{resize_width}')  # magicpoint用のデータセット
        makedirs(mpd_path1, exist_ok=True)
        mpd_path2 = join(mpd_path1, "city_scapes")
        makedirs(mpd_path2, exist_ok=True)
        mp_dirs = []
        for ip in ["images", "points"]:
            mp_dirs.append(join(mpd_path2, ip))
            makedirs(join(mpd_path2, ip), exist_ok=True)
        

        for key, value in tasks.items():
            if key == 'test': break
            # magicpoint:ディレクトリの構築
            mp_key_path = join(mp_dirs[0], value[0])
            makedirs(mp_key_path, exist_ok=True)
            mp_value_path = join(mp_dirs[1], value[0])
            makedirs(mp_value_path, exist_ok=True)
            print(f'{key}:ディレクトリの構成完了')

            print(f'{key}:magicpoint, superpoint用の画像')
            for path in tqdm(value[2]):
                # 画像の読み込み
                o_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
                mp_img = cv2.resize(o_img, (160, 120)) # magicpoint:画像の読み込みとリサイズ
                img1 = cv2.resize(o_img, (resize_width, resize_height))
                img2, center = img_rotate(img1, angle)

                # akazeの実行
                akaze = cv2.AKAZE_create()
                kp1, des1 = akaze.detectAndCompute(img1, None)
                kp2, des2 = akaze.detectAndCompute(img2, None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                # マッチングした特徴点同士の二点間距離を計算する、
                # 条件として、
                # ①アフィン変換していない画像の特徴点は何にもしない
                # ②アフィン変換した画像の特徴点は画像中心周りに時計回りにθ度回転移動させた特徴点を使用
                # ③①、②の特徴点の二点間距離を計算する
                pnts = []
                for m in matches:
                    kp_1 = kp1[m.queryIdx].pt
                    kp_2 = rotation(kp2[m.trainIdx].pt, center, angle)
                    distance = np.linalg.norm(kp_1-kp_2)
                    if distance <= NORM_THESHOLD:
                        pnts.append([kp_1[1], kp_1[0]])

                if len(pnts) != 0: 
                    pnts = np.array(pnts)
                    pnts[:, 0] = pnts[:, 0] * 120 / resize_height
                    pnts[:, 1] = pnts[:, 1] * 160 / resize_width
                    pnts = np.round(pnts, decimals=3)

                    filename = basename(path)
                    # magicpoint:リサイズした元画像の保存と選択特徴点の位置座標の保存
                    cv2.imwrite(join(mp_key_path, filename), mp_img)
                    np.save(join(mp_value_path, f'{splitext(filename)[0]}.npy'), pnts)
                
            print(f'{key}:magicpoint, superpoint用の画像の保存完了')


def ex3_angle():
    def img_rotate(img, angle):  # 画像の回転
        height = img.shape[0]
        width = img.shape[1]
        center = (int(width/2), int(height/2))
        trans = cv2.getRotationMatrix2D(center, angle, scale=1)
        image = cv2.warpAffine(img, trans, (width, height))
        return image, center
    
    def rotation(kp, r_axis, t, deg=True):

        # 度数単位の角度をラジアンに変換
        if deg == True:
            t = np.deg2rad(t)
        r_axis = np.array(r_axis)

        # 回転行列
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t),  np.cos(t)]])
        
        return np.dot(R, kp-r_axis)+r_axis
    
    img_paths = glob(join(DATA_PATH, 'CityScapes', "train", '*', '*'))
    img1 = cv2.resize(cv2.imread(img_paths[0]), (1024, 512))
    angle = 30
    img2, center = img_rotate(img1, angle)

    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    plot_match = True
    plot_before = True
    plot_after = True
    # 対応する特徴点同士を描画
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    if plot_match:
        plt.imshow(img3)
        plt.show()

    # アフィン変換していないもの
    for kp in [kp1[m.queryIdx].pt for m in matches]:
        cv2.circle(img1, (int(kp[0]), int(kp[1])), color=(255,0,0), radius=3, thickness=-1)
    if plot_before:
        plt.imshow(img1)
        plt.show()

    # アフィン変換しているもの
    for kp in [kp2[m.trainIdx].pt for m in matches]:
        cv2.circle(img2, (int(kp[0]), int(kp[1])), color=(255,0,0), radius=3, thickness=-1)
    if plot_before:
        plt.imshow(img2)
        plt.show()

    for m in matches:
        kp_1 = kp1[m.queryIdx].pt
        kp_2 = rotation(kp2[m.trainIdx].pt, center, angle)
        distance = np.linalg.norm(kp_1-kp_2)
        if distance <= 1:
            cv2.circle(img1, (int(kp_1[0]), int(kp_1[1])), color=(0,0,255), radius=3, thickness=-1)
    
    if plot_after:
        plt.imshow(img1)
        plt.show()

def ex3(dataset:str, angle: int, mag=None, resize_height=None, resize_width=None):
    def img_rotate(img, angle):  # 画像の回転
        height = img.shape[0]
        width = img.shape[1]
        center = (int(width/2), int(height/2))
        trans = cv2.getRotationMatrix2D(center, angle, scale=1)
        image = cv2.warpAffine(img, trans, (width, height))
        return image, center
    
    def rotation(kp, r_axis, t, deg=True):

        # 度数単位の角度をラジアンに変換
        if deg == True:
            t = np.deg2rad(t)
        r_axis = np.array(r_axis)

        # 回転行列
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t),  np.cos(t)]])
        
        return np.dot(R, kp-r_axis)+r_axis

    tasks = {'train': ['training', 44800], 'val': ['validation', 6400]}
    NORM_THESHOLD = 1

    print("ディレクトリの構成中")
    if dataset == "coco":
        mpd_path = join(
            DATA_PATH,
            f'ex3_magicpoint_dataset_mag:{mag}_angle:{angle}'
        )  # magicpoint用のデータセット
    elif dataset == "city":
        # magicpoint:ディレクトリの構築
        mpd_path = join(
            DATA_PATH, 
            f'ex3_magicpoint_dataset_{angle}_height{resize_height}_width_{resize_width}'
        )  # magicpoint用のデータセット
    makedirs(mpd_path, exist_ok=True)
    mpd_path2 = join(mpd_path, "city_scapes")
    makedirs(mpd_path2, exist_ok=True)
    mp_dirs = []
    for ip in ["images", "points"]:
        mp_dirs.append(join(mpd_path2, ip))
        makedirs(join(mpd_path2, ip), exist_ok=True)
    
    # COCO(Superpoint):ディレクトリの構築
    # coco_dataset_path = join(DATA_PATH, "ex3_coco")
    # makedirs(coco_dataset_path, exist_ok=True)

    for key, value in tasks.items():
        # magicpoint:ディレクトリの構築
        mp_key_path = join(mp_dirs[0], value[0])
        makedirs(mp_key_path, exist_ok=True)
        mp_value_path = join(mp_dirs[1], value[0])
        makedirs(mp_value_path, exist_ok=True)

        # COCO(Superpoint):ディレクトリの構築
        # coco_key_path = join(coco_dataset_path, f'{key}2014')
        # makedirs(coco_key_path, exist_ok=True)
        print(f'{key}:ディレクトリの構成完了')

        if dataset == "coco":
            img_paths = natsort.natsorted(
                glob(join(DATA_PATH, 'COCO', f'{key}2014', '*')))[:value[1]]
        elif dataset == "city":
            img_paths = glob(join(DATA_PATH, 'CityScapes', key, '*', '*'))
        print(f'{key}:magicpoint, superpoint用の画像')
        for path in tqdm(img_paths):
            # 画像の読み込み
            o_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            mp_img = cv2.resize(o_img, (160, 120)) # magicpoint:画像の読み込みとリサイズ
            sp_img = o_img # superpoint:画像の読み込み
            if dataset == "coco":
                resize_height = o_img.shape[0] * mag
                resize_width = o_img.shape[1] * mag
            img1 = cv2.resize(o_img, (resize_width, resize_height))
            img2, center = img_rotate(img1, angle)

            # akazeの実行
            akaze = cv2.AKAZE_create()
            kp1, des1 = akaze.detectAndCompute(img1, None)
            kp2, des2 = akaze.detectAndCompute(img2, None)
            bf = cv2.BFMatcher(NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # マッチングした特徴点同士の二点間距離を計算する、
            # 条件として、
            # ①アフィン変換していない画像の特徴点は何にもしない
            # ②アフィン変換した画像の特徴点は画像中心周りに時計回りにθ度回転移動させた特徴点を使用
            # ③①、②の特徴点の二点間距離を計算する
            pnts = []
            for m in matches:
                kp_1 = kp1[m.queryIdx].pt
                kp_2 = rotation(kp2[m.trainIdx].pt, center, angle)
                distance = np.linalg.norm(kp_1-kp_2)
                if distance <= NORM_THESHOLD:
                    pnts.append([kp_1[1], kp_1[0]])

            if len(pnts) != 0: 
                pnts = np.array(pnts)
                pnts[:, 0] = pnts[:, 0] * 120 / resize_height
                pnts[:, 1] = pnts[:, 1] * 160 / resize_width
                pnts = np.round(pnts, decimals=3)

                filename = basename(path)
                # magicpoint:リサイズした元画像の保存と選択特徴点の位置座標の保存
                cv2.imwrite(join(mp_key_path, filename), mp_img)
                np.save(join(mp_value_path, f'{splitext(filename)[0]}.npy'), pnts)

            # COCO(Superpoint):元画像の保存
            # cv2.imwrite(join(coco_key_path, filename), sp_img)
            
        print(f'{key}:magicpoint, superpoint用の画像の保存完了')


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
            kp_list[:, 0] = kp_list[:, 0] / 2  # 480 / 2 = 240pixelに変更
            kp_list[:, 1] = kp_list[:, 1] / 2  # 640 / 2 = 320pixelに変更
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
        coco_img_paths = natsort.natsorted(glob(coco_path))
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
        img = cv2.cvtColor(cv2.resize(
            img, (width*a, height*a)), cv2.COLOR_BGR2GRAY)
        akaze = cv2.AKAZE_create()
        kp = akaze.detect(img)
        if len(kp) > 0:
            kp_list = []
            for k in kp:
                # [y,x,response]responseは特徴点の強度
                kp_list.append([k.pt[1], k.pt[0], k.response])
            kp_list = np.array(kp_list)
            # height*aサイズの位置座標 / (height*a) * 120 = 120pixelに変更
            kp_list[:, 0] = kp_list[:, 0] * 120 / (height*a)
            # width*aサイズの位置座標 / (width*a) * 160 = 160pixelに変更
            kp_list[:, 1] = kp_list[:, 1] * 160 / (width*a)
            kp_list = np.round(kp_list, decimals=3)
            resize_img = cv2.resize(img, (160, 120))
            return True, resize_img, kp_list[:, :2]
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
    tasks = {'training': ["train2014", 44800], 'validation': ["val2014", 6400]}
    for key, value in tasks.items():
        coco_path = join(DATA_PATH, 'COCO', value[0], '*')
        save_img_path = join(base_path, 'images', key)
        save_pnt_path = join(base_path, 'points', key)
        coco_img_paths = natsort.natsorted(glob(coco_path))[:value[1]]
        for path in tqdm(coco_img_paths):
            basename_without_ext = splitext(basename(path))[0]
            img = cv2.imread(path)
            judge, resize_img, kp_list = detect_feature_points_with_resize(
                img, 2)  # [y,x,stregth]
            if judge:
                cv2.imwrite(
                    f'{save_img_path}/{basename_without_ext}.png', resize_img)
                np.save(f'{save_pnt_path}/{basename_without_ext}.npy', kp_list)


def main():
    args = sys.argv
    if args[1] == "ex3":
        ex3(str(args[2]), int(args[3]), int(args[4]), int(args[5]), int(args[6]))
    elif args[1] == "ex2":
        ex2()
    elif args[1] == "ex1":
        ex1()
    elif args[1] == "ex3_d":
        # ex3_dev()
        ex3_angle()


# main()
ex3_city(512, 1024)