# 特徴点抽出法SuperPoint
これは、特徴点抽出法akazeを用いて、特徴点位置座標をまとめたデータセット「akaze-coco dataset」を作成し、同じく特徴点検出法MagicPointの学習および、評価に使用するデータセットに使用および、性能比較する。

## pythonの環境について
### 必要要件
requirements.txtより確認してください。

### Pythonのインストール方法と必要なライブラリのインストール方法

[project_dir]、[newenvname]については自分で決めてください

```
cd [project_dir]
python3 -m venv [newenvname]
source [newenvname]/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Pytorchのインストール方法

例）cudaバージョン:11.5、pytorchバージョン:1.11.0の場合は
```
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu115
```

これは、説明がめんどくさいので、「cuda pytorch インストール」とインターネットで検索すれば出てくるので調べてほしい。

## データセットの作成　　
画像と特徴点位置座標をまとめたデータセットの作成を行うプログラムである。データセットの種類として、  
- synthetic shapesデータセット
- 特徴点検出法akazeを用いて、synthetic shapesデータセットの画像から特徴点検出したデータセット
- 特徴点検出法akazeを用いて、COCOデータセットの画像から特徴点検出したデータセット

の3つである。

### 必要なデータセット
- MS-COCO 2014 
    - [MS-COCO 2014 学習用データセット link](http://images.cocodataset.org/zips/train2014.zip)
    - [MS-COCO 2014 評価用データセット link](http://images.cocodataset.org/zips/val2014.zip)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)

### パスの設定
データセットのパス(DATA_PATH)と出力結果のパス(EXPER_PATH)は`setting.py`に記述してある。自分の好きな場所にセットしてそこに必要なデータセットなどを入れて管理してほしい。ちなみに、フォルダーの構造として以下のようになる。
```
datasets/ ($DATA_DIR)
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # will be automatically created
```

ダウンロードには時間がかかるため気長に待とう。

## コードの実行

### 実験1) Synthetic Shapesデータセットおよび、akaze-cocoデータセットを使用した、MagicPointの学習と評価

#### Synthetic Shapesデータセットの場合
```
python train4.py train_base configs_synth/magicpoint_original_synth_pair.yaml magicpoint_original_synth --eval
```

#### akaze-cocoデータセットの場合
```
python train4.py train_base configs_akaze/magicpoint_akaze_coco_pair.yaml magicpoint_akaze_coco --eval
```

## 参考文献
Please cite the original paper.
```
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={224--236},
  year={2018}
}
```

Please also cite our DeepFEPE paper.
```
@misc{2020_jau_zhu_deepFEPE,
Author = {You-Yi Jau and Rui Zhu and Hao Su and Manmohan Chandraker},
Title = {Deep Keypoint-Based Camera Pose Estimation with Geometric Constraints},
Year = {2020},
Eprint = {arXiv:2007.15122},
}
```

# クレジット
This implementation is developed by [You-Yi Jau](https://github.com/eric-yyjau) and [Rui Zhu](https://github.com/Jerrypiglet).
Again the work is based on Tensorflow implementation by [Rémi Pautrat](https://github.com/rpautrat) and [Paul-Edouard Sarlin](https://github.com/Skydes) and official [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).
Thanks to Daniel DeTone for help during the implementation.
