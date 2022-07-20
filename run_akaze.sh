python create_datasets/akaze_datasets.py
sleep 10s
python train4.py train_base configs_akaze/magicpoint_akaze_coco_pair.yaml magicpoint_akaze_coco --eval
sleep 10s
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:100 python export.py export_detector_homoAdapt configs_akaze/magicpoint_akaze_coco_export_train.yaml magicpoint_akaze_coco
sleep 10s
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:100 python export.py export_detector_homoAdapt configs_akaze/magicpoint_akaze_coco_export_val.yaml magicpoint_akaze_coco
sleep 10s
python search_nan_and_delete.py
sleep 10s
python train4.py train_joint configs_akaze/superpoint_akaze_coco_train_heatmap.yaml superpoint_akaze_coco --eval --debug
sleep 10s
python export.py export_descriptor  configs_akaze/superpoint_akaze_repeatability_heatmap.yaml superpoint_akaze_hpatches_test
sleep 10s
python evaluation.py ../logs/superpoint_akaze_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching