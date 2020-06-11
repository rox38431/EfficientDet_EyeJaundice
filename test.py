import datetime
import time
import os
import argparse
import traceback
import cv2
import glob
import math
import random
import yaml
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.dataset import EyeDataset, Resizer, Normalizer, Augmenter, collater
from backbone import EfficientDetBackbone

from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, init_weights

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('EfficientDet for eye jaundiance')
    parser.add_argument('-c', '--compound_coef', type=int, default=0)
    parser.add_argument('-n', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--es_min_delta', type=float, default=0.0)
    parser.add_argument('--es_patience', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('-w', '--weight_path', type=str, default=None)
    parser.add_argument('--saved_path', type=str, default='logs')
    parser.add_argument('--dataset_path', type=str, default='./../eye_data/raw')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = FocalLoss()
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward(self, imgs, annotations, obj_list=None):
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


def main(args):
    print("Hi")
    assert args.weight_path, 'must indicate the path of pre-trained weight'
    params = Params(f'projects/eye.yml')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'drop_last': True,
                   'collate_fn': collater,
                   'num_workers': args.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    model = EfficientDetBackbone(num_classes=len(params.obj_list),
                                 compound_coef=args.compound_coef,
                                 ratios=eval(params.anchors_ratios),
                                 scales=eval(params.anchors_scales))
    init_weights(model)
    model = ModelWithLoss(model)
    model = model.cuda()

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    model.model.load_state_dict(torch.load(f'{args.weight_path}/pre_trained_weight.pth')['model_state_dict'])
    optimizer.load_state_dict(torch.load(f'{args.weight_path}/pre_trained_weight.pth')['optimizer_state_dict'])
    scheduler.load_state_dict(torch.load(f'{args.weight_path}/pre_trained_weight.pth')['scheduler_state_dict'])

    test_img_list = glob.glob(f'{args.dataset_path}/test/*')
    test_anno_txt_path = f'{args.dataset_path}/test.txt'
    
    test_transform = transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                         Augmenter(),
                                         Resizer(input_sizes[args.compound_coef])])

    test_set = EyeDataset(test_img_list, test_anno_txt_path, test_transform)
    test_generator = DataLoader(test_set, **test_params)

    model.eval()
    with torch.no_grad():
        total = 0
        total_correct = 0
        total_loss_ls = []
        for data in test_generator:
            imgs = data['img'].cuda()
            annot = data['annot'].cuda()

            reg_loss, cls_head_loss, cls_correct_num, total_num = model(imgs, annot, obj_list=params.obj_list)

            total_correct += cls_correct_num
            total += total_num
            reg_loss = reg_loss.mean()
            loss = reg_loss + cls_head_loss
            total_loss_ls.append(loss.item())
        total_loss = np.mean(total_loss_ls)
        print(f'Testing loss: {total_loss:.6f} | acc: {total_correct / total * 100:.2f}')


if __name__ == '__main__':
    args = get_args()
    main(args)
