import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
from tqdm import tqdm

import coco_eval
import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', default='coco', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', default='/home/hao.wyh/jupyter/黑边/smart_reverse_label/coco/', help='Path to COCO directory')
    #parser.add_argument('--coco_path', default='/home/hao.wyh/jupyter/黑边/评估任务/3k_imgs/coco/', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--model', help='Path to model (.pt) file.')
    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = torch.load(parser.model)
    use_gpu = True
    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()
    coco_eval.evaluate_coco(dataset_val, retinanet)
if __name__ == '__main__':
    main()
