import glob
import cv2
import numpy as np
from tqdm import tqdm
from utils.photometric import ImgAugTransform, customizedTransform
from imageio import imread

DATA_PATH = '/Users/gennosuke/Downloads/datasets'

configs = {
    "augmentation": {
        "photometric": {
            "enable": True,  # for class to recognize
            "enable_train": True,
            "enable_val": False,
            "primitives": [
                'random_brightness', 'random_contrast', 'additive_speckle_noise',
                'additive_gaussian_noise', 'additive_shade', 'motion_blur'],
            "params": {
                "random_brightness": {"max_abs_change": 75},
                "random_contrast": {"strength_range": [0.3, 1.8]},
                "additive_gaussian_noise": {"stddev_range": [0, 15]},
                "additive_speckle_noise": {"prob_range": [0, 0.0035]},
                "additive_shade": {
                    "transparency_range": [-0.5, 0.8],
                    "kernel_size_range": [50, 100]
                },
                "motion_blur": {"max_kernel_size": 7}  # origin 7
            }
        },
        "homographic": {
            "enable": True,
            "enable_train": True,
            "enable_val": False,
            "params": {
                "translation": True,
                "rotation": True,
                "scaling": True,
                "perspective": True,
                "scaling_amplitude": 0.2,
                "perspective_amplitude_x": 0.2,
                "perspective_amplitude_y": 0.2,
                "patch_ratio": 0.8,
                "max_angle": 1.57,  # 3.14
                "allow_artifacts": True,
                "translation_overflow": 0.05,
            },
            "valid_border_margin": 2
        }
    }
}

def load_as_float(path):
    return imread(path).astype(np.float32) / 255

def imgPhotometric(img, config):
    """

    :param img:
        numpy (H, W)
    :return:
    """
    augmentation = ImgAugTransform(**config["augmentation"])
    img = img[:, :, np.newaxis]
    img = augmentation(img)
    cusAug = customizedTransform()
    img = cusAug(img, **config["augmentation"])
    return img

train_paths = glob.glob(DATA_PATH + '/akaze_coco/ms-coco/images/training/*')
val_paths = glob.glob(DATA_PATH + '/akaze_coco/ms-coco/images/validation/*')

for i in tqdm(range(len(train_paths))):
    img = load_as_float(train_paths[i])
    img = imgPhotometric(img, configs)

print("Test OK!!")
