# import math
from os.path import join

import numpy as np

LOGS_PATH = '/Users/gennosuke/Downloads/logs'
akaze_150_result_npz_path = join(
    LOGS_PATH,
    'superpoint_akaze_hpatches_test',
    '200000_150_superpoint_akaze_npz',
    'result.npz')

akaze_200_result_npz_path = join(
    LOGS_PATH,
    'superpoint_akaze_hpatches_test',
    '200000_200_superpoint_akaze_npz',
    'result.npz')

original_result_npz_path = join(
    LOGS_PATH,
    'superpoint_original_hpatches_test',
    'predictions',
    'result.npz')

akaze_150_npz = np.load(akaze_150_result_npz_path)
akaze_200_npz = np.load(akaze_200_result_npz_path)
original_npz = np.load(original_result_npz_path)
print(akaze_150_npz.files)
print(sum(akaze_150_npz["mscore"])/len(akaze_150_npz))
print(sum(akaze_200_npz["mscore"])/len(akaze_200_npz))
print(sum(original_npz["mscore"])/len(original_npz))
print(akaze_150_npz["n_matches_in"])
print(akaze_200_npz["n_matches_in"])
print(original_npz["n_matches_in"])

cnt = 0
for aka, ori in zip(akaze_150_npz["n_matches_in"], original_npz["n_matches_in"]):
    if aka > ori:
        cnt += 1
print(cnt)
cnt = 0
for aka, ori in zip(akaze_200_npz["n_matches_in"], original_npz["n_matches_in"]):
    if aka > ori:
        cnt += 1
print(cnt)
# cnt = 0
# for aka, ori in zip(akaze_150_npz["mscore"], original_npz["mscore"]):
#     if aka > ori:
#         print(cnt)
#     cnt += 1

# cnt = 0
# for aka, ori in zip(akaze_200_npz["mscore"], original_npz["mscore"]):
#     if aka > ori:
#         print(cnt)
#     cnt += 1