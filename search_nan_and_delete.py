import numpy as np
from glob import glob

train_val_npy_path = [
    '/home/gennosuke/logs/magicpoint_ms_coco_akaze/predictions/train/*',
    '/home/gennosuke/logs/magicpoint_ms_coco_akaze/predictions/val/*'
]
error_npz_path = []

for npz_path in train_val_npy_path:
    path_list = glob(npz_path)
    for path in path_list[:]:
        labels = np.zeros((240, 320))
        npy = np.load(path)
        pnts = npy['pts']
        pnts = pnts.astype(int)
        try:
            labels[pnts[:, 1], pnts[:, 0]] = 1
        except IndexError as e:
            error_npz_path.append(path)

for ep in error_npz_path:
    error_npy_file = np.load(ep)
    npz = error_npy_file['pts']
    new_array = []
    for n in npz:
        if np.isnan(n[0]) or np.isnan(n[1]):
            continue
        else:
            new_array.append(n)
    new_array = np.array(new_array)
    np.savez(ep, pts=new_array)