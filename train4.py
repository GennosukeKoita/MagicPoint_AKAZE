"""Training script
This is the training script for superpoint detector and descriptor.

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

from utils.loader import get_save_path
import argparse
import yaml
import os
import logging

import torch
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter
from utils.utils import getWriterPath
from settings import EXPER_PATH

# loaders: data, model, pretrained model
from utils.loader import dataLoader
from utils.logging import *

###### util functions ######


def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches' %
                 (tag, len(train_loader)*config['model']['batch_size'], len(train_loader)))
    pass


###### util functions end ######


###### train script ######
def train_base(config, output_dir, args):
    return train_joint(config, output_dir, args)


def train_joint(config, output_dir, args):
    assert 'train_iter' in config
    
    # torch.set_default_tensor_type：テンソルを作成する際にデバイスを指定しないときに使う.デフォルトはcpu,gpuが積んでいる場合はgpuになる。
    torch.set_default_tensor_type(torch.FloatTensor)
    task = config['data']['dataset']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    # tensorboard.summaryWriter:グラフの描画に使用するもの.
    writer = SummaryWriter(
        getWriterPath(task=args.command, exper_name=args.exper_name, date=True))
    # checkpointを保存するディレクトリのパスが返される
    save_path = get_save_path(output_dir)

    # 学習、評価データのロード
    data = dataLoader(config, dataset=task, warp_input=True)
    train_loader, val_loader = data['train_loader'], data['val_loader']
    datasize(train_loader, config, tag='train')
    datasize(val_loader, config, tag='val')

    # init the training agent using config file
    from utils.loader import get_module
    train_model_frontend = get_module('', config['front_end_model'])
    train_agent = train_model_frontend(
        config, save_path=save_path, device=device)

    # writer from tensorboard
    train_agent.writer = writer
    # feed the data into the agent
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader
    # load model initiates the model and load the pretrained model (if any)
    train_agent.loadModel()
    train_agent.dataParallel()

    try:
        # train function takes care of training and evaluation
        train_agent.train()
    except KeyboardInterrupt:
        print("press ctrl + c, save model!")
        train_agent.saveModel()
        pass


if __name__ == '__main__':
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training magicpoint command
    p_train = subparsers.add_parser('train_base')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=train_base)

    # Training superpoint command
    p_train = subparsers.add_parser('train_joint')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=train_joint)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            format='[%(asctime)s %(levelname)s] %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # EXPER_PATH from settings.py
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Running command {}'.format(args.command.upper()))
    args.func(config, output_dir, args)
