import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):


    coco_path = '/home/hao.wyh/jupyter/黑边/x小视频封面图/made/csv/coco/coco/'
    coco_path = '/home/hao.wyh/jupyter/黑边/smart_reverse_label/coco'


    dataset_train = CocoDataset(coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=0, batch_size=1, collate_fn=collater)#, batch_sampler=sampler)

    # dataset_val = CocoDataset(coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    # dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    # retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)	
    # retinanet = retinanet.cuda()
    # retinanet.training = True
    # optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # loss_hist = collections.deque(maxlen=500)
    # retinanet.train()
    # retinanet.freeze_bn()
    # print('Num training images: {}'.format(len(dataset_train)))
    # epoch_loss = []
    for i in (0,1):
        data = dataset_train[i]
        print(data.keys())
        print(data['img'].shape)
        print(data['annot'])
        print(data['sc_h'])
        print(data['sc_w'])
    for iter_num, data in enumerate(dataloader_train):
        try:
            # optimizer.zero_grad()
            if iter_num < 2:
                print(data['img'].shape)
                print(data['annot'])
            else:
                break
            # classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

        except Exception as e:
            print(e)
            continue

    # print('Evaluating dataset')
    # coco_eval.evaluate_coco(dataset_val, retinanet)

    # scheduler.step(np.mean(epoch_loss))	
    # torch.save(retinanet.module, os.path.join(output_path, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num)))
    # retinanet.eval()
    # torch.save(retinanet, os.path.join(output_path, 'model_final.pt'.format(epoch_num)))

if __name__ == '__main__':
    main()



    def __call__(self, sample, min_side=608, max_side=1024):
        min_side, max_side = 300, 600
        image, annots = sample['img'], sample['annot']

        sh, sw, cns = image.shape
        h, w, cns = image.shape

        smallest_side = min(h, w)
        scale = min_side / smallest_side
        largest_side = max(h, w)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        h, w, cns = int(round(h*scale)), int(round(w*scale)), cns

        pad_w = 32 - h%32
        pad_h = 32 - w%32

        new_w = w + pad_w
        new_h = h + pad_h

        scale_w = new_w / sw
        scale_h = new_h / sh
        new_image = skimage.transform.resize(image, (new_w, new_h), mode='constant').astype(np.float32)

        annots[:, 0] *= scale_w
        annots[:, 1] *= scale_h
        annots[:, 2] *= scale_w
        annots[:, 3] *= scale_h

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'sc_w':scale_w,'sc_h':scale_h}

    def __call__(self, sample, min_side=608, max_side=1024):
        min_side, max_side = 300, 600
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape
        srows, scols, scns = image.shape

        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        rows, cols, cns = int(round(rows*scale)), int(round(cols*scale)), cns

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_w = rows + pad_w
        new_h = cols + pad_h

        scale_w = new_w / srows
        scale_h = new_h / scols
        new_image = skimage.transform.resize(image, (new_w, new_h), mode='constant').astype(np.float32)

        annots[:, 0] *= scale_w
        annots[:, 1] *= scale_h
        annots[:, 2] *= scale_w
        annots[:, 3] *= scale_h

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'sc_w':scale_w,'sc_h':scale_h}

