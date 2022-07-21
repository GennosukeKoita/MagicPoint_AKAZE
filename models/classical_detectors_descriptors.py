import sys
import tensorflow as tf
import numpy as np
import cv2

# from .base_model import BaseModel
# from .utils import box_nms

def classical_detector_descriptor(im, method, detection_threshold):
    im = np.uint8(im)
    if method == 'sift':
        sift = cv2.SIFT_create(nfeatures=1500)
        keypoints, desc = sift.detectAndCompute(im, None)
        responses = np.array([k.response for k in keypoints])
        keypoints = np.array([k.pt for k in keypoints]).astype(int)
        desc = np.array(desc)

        detections = np.zeros(im.shape[:2], np.float)
        detections[keypoints[:, 1], keypoints[:, 0]] = responses
        descriptors = np.zeros((im.shape[0], im.shape[1], 128), np.float)
        descriptors[keypoints[:, 1], keypoints[:, 0]] = desc
    
    elif method == 'akaze':
        akaze = cv2.AKAZE_create()
        keypoints, desc = akaze.detectAndCompute(im, None)
        responses = np.array([k.response for k in keypoints])
        keypoints = np.array([k.pt for k in keypoints]).astype(int)
        desc = np.array(desc)
        new_keypoints = []
        new_responses = []
        new_desc = []
        for key, des, res in zip(keypoints, desc, responses):
            if res >= detection_threshold:
                new_keypoints.append(key)
                new_responses.append(res)
                new_desc.append(des)
        keypoints = np.array(new_keypoints)
        responses = np.array(new_responses)
        desc = np.array(new_desc)
        detections = np.zeros((im.shape[:2]), np.float64)
        descriptors = np.zeros((im.shape[0], im.shape[1], 61), np.float64)
        if responses != []:
            detections[keypoints[:, 1], keypoints[:, 0]] = responses
            descriptors[keypoints[:, 1], keypoints[:, 0]] = desc

    elif method == 'orb':
        orb = cv2.ORB_create(nfeatures=1500)
        keypoints, desc = orb.detectAndCompute(im, None)
        responses = np.array([k.response for k in keypoints])
        keypoints = np.array([k.pt for k in keypoints]).astype(int)
        desc = np.array(desc)

        detections = np.zeros(im.shape[:2], np.float)
        detections[keypoints[:, 1], keypoints[:, 0]] = responses
        descriptors = np.zeros((im.shape[0], im.shape[1], 32), np.float)
        descriptors[keypoints[:, 1], keypoints[:, 0]] = desc

    detections = detections.astype(np.float32)
    descriptors = descriptors.astype(np.float32)
    return (detections, descriptors)

# from models.classical_detector_descriptors import SIFT_det
def SIFT_det(img, img_rgb, visualize=False, nfeatures=2000):
    """
    return: 
        x_all: np [N, 2] (x, y)
        des: np [N, 128] (descriptors)
    """
    # Initiate SIFT detector
    # pip install opencv-python==3.4.2.16, opencv-contrib-python==3.4.2.16
    # https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
    img = np.uint8(img)
    # print("img: ", img)
    sift = cv2.SIFT_create(contrastThreshold=1e-5)

    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img, None)
    # print("# kps: {}, descriptors: {}".format(len(kp), des.shape))
    x_all = np.array([p.pt for p in kp])

    if visualize:
        plt.figure(figsize=(30, 4))
        plt.imshow(img_rgb)

        plt.scatter(x_all[:, 0], x_all[:, 1], s=10, marker='o', c='y')
        plt.show()

    # return x_all, kp, des

    return x_all, des