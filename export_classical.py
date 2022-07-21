"""export classical feature extractor (not tested)
"""

import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
from settings import EXPER_PATH
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.utils import getWriterPath

# from utils.logging import *


def export_descriptor(config, output_dir, args):
    '''
    1) input 2 images, output keypoints and correspondence

    :param config:
    :param output_dir:
    :param args:
    :return:
    '''
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(getWriterPath(task=args.command, date=True))

    # save data
    from pathlib import Path
    # save_path = save_path_formatter(config, output_dir)
    save_path = Path(output_dir)
    save_output = save_path
    save_output = save_output / 'predictions'
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_output, exist_ok=True)

    # data loading
    from utils.loader import dataLoader_test as dataLoader
    data = dataLoader(config, dataset='hpatches')
    test_set, test_loader = data['test_set'], data['test_loader']

    from utils.print_tool import datasize
    datasize(test_loader, config, tag='test')
    
    def squeezeToNumpy(tensor_arr):
        return tensor_arr.detach().cpu().numpy().squeeze()

    count = 0
    method = config['model']['method']

    for i, sample in tqdm(enumerate(test_loader)):

        img_0, img_1 = sample['image'], sample['warped_image']
        imgs_np, imgs_fil = [], []
        # first image, no matches
        imgs_np.append(img_0.numpy().squeeze())
        imgs_np.append(img_1.numpy().squeeze())

        ##### add opencv functions here #####
        def classicalDetectors(image, method='sift'):
            """
            # sift keyframe detectors and descriptors
            """
            image = image*255
            round_method = True
            if round_method == True:
                # with quantization
                from models.classical_detectors_descriptors import classical_detector_descriptor
                detection_threshold = config['model']['detection_threshold'] if config[
                    'model']['detection_threshold'] != '' else 0
                points, desc = classical_detector_descriptor(
                    image, method, detection_threshold)
                if np.all(points == 0.):
                    return points, desc
                else:
                    y, x = np.where(points)
                    # pnts = np.stack((y, x), axis=1)
                    pnts = np.stack((x, y), axis=1)  # should be (x, y)
                    # collect descriptros
                    desc = desc[y, x, :]
            else:
                # sift with subpixel accuracy
                from models.classical_detectors_descriptors import SIFT_det as classical_detector_descriptor
                pnts, desc = classical_detector_descriptor(image, image)

            print("desc shape: ", desc.shape)
            return pnts, desc

        pts_list = []
        pts, desc_1 = classicalDetectors(imgs_np[0], method=method)
        if np.all(pts == 0.):
            continue
        pts_list.append(pts)
        print("total points: ", pts.shape)
        '''
        pts: list [numpy (N, 2)]
        desc: list [numpy (N, 128)]
        '''
        # save keypoints
        pred = {}
        pred.update({
            'image': imgs_np[0],
        })
        pred.update({'prob': pts,
                     'desc': desc_1})

        # second image, output matches

        pred.update({
            'warped_image': imgs_np[1],
        })
        pts, desc_2 = classicalDetectors(imgs_np[1], method=method)
        if np.all(pts == 0.):
            continue
        pts_list.append(pts)

        print("total points: ", pts.shape)
        pred.update({'warped_prob': pts,
                     'warped_desc': desc_2,
                     'homography': squeezeToNumpy(sample['homography'])
                     })

        # get matches
        match_judge, data = get_match(kps_ii=pts_list[0], des_ii=desc_1, kps_jj=pts_list[1], des_jj=desc_2, if_BF_matcher=True)
        if not match_judge:
            continue
        matches = data['match_quality_good']
        print(f"matches: {matches.shape}")
        matches_all = data['match_quality_all']
        pred.update({
            'matches': matches,
            'matches_all': matches_all
        })
        # clean last descriptor
        '''
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N2, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)
            'matches': np (N3, 4)
        '''

        # save data
        from pathlib import Path
        filename = str(count)
        path = Path(save_output, '{}.npz'.format(filename))
        np.savez_compressed(path, **pred)
        count += 1

    print("output pairs: ", count)
    save_file = save_output / "export.txt"
    with open(save_file, "a") as myfile:
        myfile.write("output pairs: " + str(count) + '\n')
    pass


def get_match(kps_ii, des_ii, kps_jj, des_jj, if_BF_matcher=True):
    # select which kind of matcher
    if (
        if_BF_matcher
    ):  # OpenCV matcher must be created inside each thread (because it does not support sharing across threads!)
        bf = cv2.BFMatcher(normType=cv2.NORM_L2)
        matcher = bf
    else:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matcher = flann

    n_match_judge, all_ij, good_ij, quality_good, quality_all, cv_matches = get_match_idx_pair(
        matcher, des_ii.copy(), des_jj.copy()
    )

    if not n_match_judge or good_ij.shape[0] == 0:
        return False, {'match_quality_good': 0, 'match_quality_all': 0, 'cv_matches': 0}
    # dump_ij_idx_file = dump_dir / "ij_idx_{}-{}".format(ii, jj)
    # dump_ij_quality_file = dump_dir / "ij_quality_{}-{}".format(ii, jj)
    # dump_ij_match_quality_file = dump_dir / "ij_match_quality_{}-{}".format(ii, jj)

    match_quality_good = np.hstack(
        (kps_ii[good_ij[:, 0]], kps_jj[good_ij[:, 1]], quality_good)
    )  # [[x1, y1, x2, y2, dist_good, ratio_good]]
    match_quality_all = np.hstack(
        (kps_ii[all_ij[:, 0]], kps_jj[all_ij[:, 1]], quality_all)
    )  # [[x1, y1, x2, y2, dist_good, ratio_good]]
    return True, {'match_quality_good': match_quality_good, 'match_quality_all': match_quality_all, 'cv_matches': cv_matches}



def get_match_idx_pair(matcher, des1, des2):
    """
    do matchings, test the quality of matchings
    """
    matches = matcher.knnMatch(
        des1, des2, k=2
    )  # another option is https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py#L309
    if len(matches) <= 1:
        return False, None, None, None, None, None
    # except Exception as e:
        # logging.error(traceback.format_exception(*sys.exc_info()))
        # return False, None, None, None, None, None
    # store all the good matches as per Lowe's ratio test.
    good = []
    all_m = []
    quality_good = []
    quality_all = []
    for m, n in matches:
        all_m.append(m)
        if m.distance < 0.8 * n.distance:
            good.append(m)
            quality_good.append([m.distance, m.distance / n.distance])
        quality_all.append([m.distance, m.distance / n.distance])

    good_ij = [[mat.queryIdx for mat in good], [mat.trainIdx for mat in good]]
    all_ij = [[mat.queryIdx for mat in all_m], [mat.trainIdx for mat in all_m]]
    return (
        True,
        np.asarray(all_ij, dtype=np.int32).T.copy(),
        np.asarray(good_ij, dtype=np.int32).T.copy(),
        np.asarray(quality_good, dtype=np.float32).copy(),
        np.asarray(quality_all, dtype=np.float32).copy(),
        matches
    )



if __name__ == '__main__':
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # export command
    p_train = subparsers.add_parser('export_descriptor')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    # p_train.add_argument('exper', type=str)
    p_train.add_argument('--correspondence', action='store_true')
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=export_descriptor)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info('Running command {}'.format(args.command.upper()))
    args.func(config, output_dir, args)

    # global variables

    # main()
