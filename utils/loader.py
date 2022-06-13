"""many loaders
# loader for model, dataset, testing dataset
"""

import os
import logging
from pathlib import Path
import numpy as np
import torch
import torch.optim
import torch.utils.data
from utils.utils import load_checkpoint

def get_save_path(output_dir):
    """
    This func
    :param output_dir:
    :return:
    """
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def worker_init_fn(worker_id):
   """The function is designed for pytorch multi-process dataloader.
   Note that we use the pytorch random generator to generate a base_seed.
   Please try to be consistent.

   References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

   """
   base_seed = torch.IntTensor(1).random_().item()
   # print(worker_id, base_seed)
   np.random.seed(base_seed + worker_id)


def dataLoader(config, dataset='syn', warp_input=False, train=True, val=True):
    # transforms:データに対して行う前処理を行うオブジェクト
    import torchvision.transforms as transforms
    training_params = config.get('training', {}) # .get(キー, 戻り値):辞書のキーが存在しなかった場合にある戻り値を返す
    workers_train = training_params.get('workers_train', 1) # 16
    workers_val   = training_params.get('workers_val', 1) # 16
    
    logging.info(f"workers_train: {workers_train}, workers_val: {workers_val}")
    # Compose():　複数の処理を行うための関数 詳しくはhttps://pystyle.info/pytorch-list-of-transforms/#outline__3で
    # ToTensor(): 正規化（例：[0,255]→[0,1.0]）.データの型もunit8からfloat32に変わる
    data_transforms = {
        'train': transforms.Compose([transforms.ToTensor()]),
        'val': transforms.Compose([transforms.ToTensor()]),
    }
    Dataset = get_module('datasets', dataset)
    print(f"dataset: {dataset}")

    train_set = Dataset(transform=data_transforms['train'], task = 'train', **config['data'],)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config['model']['batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_train,
        worker_init_fn=worker_init_fn
    )
    val_set = Dataset(transform=data_transforms['train'], task = 'val', **config['data'],)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config['model']['eval_batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_val,
        worker_init_fn=worker_init_fn
    )
    return {'train_loader': train_loader, 'val_loader': val_loader,
            'train_set': train_set, 'val_set': val_set}

def dataLoader_test(config, dataset='syn', export_task='train'):
    import torchvision.transforms as transforms
    training_params = config.get('training', {})
    workers_test = training_params.get('workers_test', 1) # 16
    logging.info(f"workers_test: {workers_test}")

    data_transforms = {'test': transforms.Compose([transforms.ToTensor()])}
    test_loader = None
    if dataset == 'hpatches':
        from datasets.patches_dataset import PatchesDataset
        if config['data']['preprocessing']['resize']:
            size = config['data']['preprocessing']['resize']
        test_set = PatchesDataset(transform=data_transforms['test'], **config['data'])
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn
        )
    else:
        logging.info(f"load dataset from : {dataset}")
        Dataset = get_module('datasets', dataset)
        test_set = Dataset(export=True, task=export_task, **config['data'])
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn
        )
    return {'test_set': test_set, 'test_loader': test_loader}

def get_module(path, name):
    import importlib
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)

def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)

def modelLoader(model='SuperPointNet', **options):
    # create model
    logging.info("=> creating model: %s", model)
    net = get_model(model)
    net = net(**options)
    return net


# mode: 'full' means the formats include the optimizer and epoch
# full_path: if not full path, we need to go through another helper function
def pretrainedLoader(net, optimizer, epoch, path, mode='full', full_path=False):
    # load checkpoint
    if full_path == True:
        checkpoint = torch.load(path)
    else:
        checkpoint = load_checkpoint(path)
    # apply checkpoint
    if mode == 'full':
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['n_iter']
    else:
        net.load_state_dict(checkpoint)
    return net, optimizer, epoch

if __name__ == '__main__':
    net = modelLoader(model='SuperPointNet')