import datetime
import time
import os
import argparse
import traceback
import cv2
import glob
import math
import random

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.dataset import EyeDataset, Resizer, Normalizer, Augmenter, collater
from backbone import EfficientDetBackbone
# from tensorboardX import SummaryWriter
import numpy as np
# from tqdm.autonotebook import tqdm

from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('EfficieitDet for eye jaundiance (cross validation)')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu choosed to run the code')
    parser.add_argument('--saved_path', type=str, default='logs')
    parser.add_argument('--dataset_path', type=str, default='./../eye_data/raw')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward(self, imgs, annotations, obj_list=None):  # Annotations is gt
        features, regression, cls_result, anchors = self.model(imgs)
        cls_target = annotations[:, 0, -1]
        cls_target = cls_target.type(torch.LongTensor)
        cls_target = cls_target.cuda()
        
        reg_loss = self.criterion(regression, anchors, annotations)
        cls_head_loss = self.cls_criterion(cls_result, cls_target)

        _, predicted = cls_result.max(1)
        total_num = cls_target.size(0)
        cls_correct_num = predicted.eq(cls_target).sum().item()
        return reg_loss, cls_head_loss, cls_correct_num, total_num


def prepare_dir(opt, present_time):
    opt.saved_path = f"./{opt.saved_path}/{present_time}"
    os.makedirs(opt.saved_path, exist_ok=False)


def train(opt):
    print("Hi")
    present_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    params = Params(f'projects/eye.yml')

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    prepare_dir(opt, present_time)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    '''
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        
        print('[Info] initializing weights...')
        init_weights(model)
    '''
    init_weights(model)

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model)
    model = model.cuda()

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)  # unit is epoch
    
    torch.save({
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        }, f"{opt.saved_path}/init_weight.pth")

    k = 10
    train_img_list = glob.glob(f"{opt.dataset_path}/train/*")
    random.shuffle(train_img_list)
    part_num = math.ceil(len(train_img_list) / k)
    last_acc = []
    last_loss = []
    for i in range(k):
        best_loss = 1e5

        model.model.load_state_dict(torch.load(f"{opt.saved_path}/init_weight.pth")["model_state_dict"])
        optimizer.load_state_dict(torch.load(f"{opt.saved_path}/init_weight.pth")["optimizer_state_dict"])
        scheduler.load_state_dict(torch.load(f"{opt.saved_path}/init_weight.pth")["scheduler_state_dict"])
        model.train()

        sub_train_img_list = train_img_list[:i*part_num] + train_img_list[(i+1)*part_num:]
        sub_test_img_list = train_img_list[i*part_num:(i+1)*part_num]

        train_anno_txt_path = f"{opt.dataset_path}/train.txt"
        test_anno_txt_path = f"{opt.dataset_path}/train.txt"

        train_transform = transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                    Augmenter(),
                                    Resizer(input_sizes[opt.compound_coef])])
        test_transform = transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                             Augmenter(),
                                             Resizer(input_sizes[opt.compound_coef])])

        train_set = EyeDataset(sub_train_img_list, train_anno_txt_path, train_transform)
        test_set = EyeDataset(sub_test_img_list, test_anno_txt_path, test_transform)
        training_generator = DataLoader(train_set, **training_params)
        val_generator = DataLoader(test_set, **val_params)

        for epoch in range(opt.epoch):
            model.train()
            total_correct = 0
            total = 0
            total_loss_ls = []
            for data in training_generator:
                imgs = data['img']
                annot = data['annot']

                imgs = imgs.cuda()
                annot = annot.cuda()

                optimizer.zero_grad()
                reg_loss, cls_head_loss, cls_correct_num, total_num = model(imgs, annot, obj_list=params.obj_list)
                total_correct += cls_correct_num
                total += total_num
                reg_loss = reg_loss.mean()
                loss = cls_head_loss + reg_loss
                total_loss_ls.append(loss.item())

                if loss == 0 or not torch.isfinite(loss):
                    continue

                loss.backward()
                optimizer.step()
            total_loss = np.mean(total_loss_ls)
            scheduler.step(total_loss)
            with open(f'./logs/{present_time}/cv_log.txt', 'a') as fp:
                fp.write(f"Epoch: {i}/{epoch}/{opt.epoch}\n")
                fp.write(f"Training loss: {total_loss:.6f} | acc: {total_correct / total * 100:.2f}\n")

            model.eval()
            with torch.no_grad():
                total = 0
                total_correct = 0
                total_loss_ls = []
                for data in val_generator:
                    imgs = data['img'].cuda()
                    annot = data['annot'].cuda()
                    
                    reg_loss, cls_head_loss, cls_correct_num, total_num = model(imgs, annot, obj_list=params.obj_list)
                    total_correct += cls_correct_num
                    total += total_num
                    reg_loss = reg_loss.mean()
                    loss = reg_loss + cls_head_loss
                    total_loss_ls.append(loss.item())
                total_loss = np.mean(total_loss_ls)
                
                with open(f'./logs/{present_time}/cv_log.txt', 'a') as fp:
                    fp.write(f"Testing loss: {total_loss:.6f} | acc: {total_correct / total * 100:.2f}\n\n")

                if (epoch == opt.epoch-1):
                    last_loss.append(total_loss)
                    last_acc.append(total_correct / total *100)

    with open(f'./logs/{present_time}/cv_log.txt', 'a') as fp:
        fp.write("\n===========\n\n")
        fp.write(f"Avg. loss: {np.mean(np.array(last_loss)):.2f}\n")
        fp.write(f"Avg. accuracy: {np.mean(np.array(last_acc)):.2f}\n")


if __name__ == '__main__':
    opt = get_args()
    train(opt)
