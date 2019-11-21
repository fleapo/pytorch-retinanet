import os
from dataloader import Resizer, Normalizer
from torchvision import datasets, models, transforms
import cv2
import numpy as np
from glob import glob
import torch
import argparse
from tqdm import tqdm


def main(args=None):
    transform=transforms.Compose([Normalizer(), Resizer()])
    annot = np.array([[10,10,20,20,0],], dtype=np.float64)

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    #parser.add_argument('--model', help='Path to model (.pt) file.', default='/home/hao.wyh/code/git/pytorch-retinanet/output_models/main_detect_v11_restart/model_final.pt')
    parser.add_argument('--model', help='Path to model (.pt) file.', default='/home/hao.wyh/code/git/pytorch-retinanet/output_models/true_data_v2_mix/coco_retinanet_3.pt')
    #parser.add_argument('--model', help='Path to model (.pt) file.', default='output_models/main_detect_v10_deeper8/model_final.pt')
    parser.add_argument('--output_path', help='Path to save output imgs.')
    #parser.add_argument('--input_path', help='Path to save output imgs.', default='/home/hao.wyh/jupyter/黑边/smart_reverse_label')
    #parser.add_argument('--input_path', help='Path to save output imgs.', default='/home/hao.wyh/jupyter/黑边/评估任务/black_imgs')
    parser.add_argument('--input_path', help='Path to save output imgs.', default='/home/hao.wyh/jupyter/黑边/评估任务/3k_imgs')
    parser = parser.parse_args(args)

    retinanet = torch.load(parser.model)
    retinanet = retinanet.cuda()
    retinanet.eval()

    ll = glob(parser.input_path+'/*jpg')
    if len(ll) == 0:
        ll = glob(parser.input_path+'/*jpeg')
    if not os.path.exists(parser.output_path):
        os.mkdir(parser.output_path)

    res = []
    for i in tqdm(ll):
        name = i.split('/')[-1]
        im = cv2.imread(i)
        img = im.astype(np.float32)/255.0
        samp = {'img':img, 'annot':annot}
        model_input = transform(samp)
        img = model_input['img'].cuda().float().unsqueeze(0).permute((0,3,1,2)).contiguous()
        sc_h = model_input['sc_h']
        sc_w = model_input['sc_w']
        #print(img.shape, sc_w, sc_h)
        scores, classification, transformed_anchors = retinanet(img)
        anchors = transformed_anchors.clone()
        anchors[:,0] /= sc_w
        anchors[:,1] /= sc_h
        anchors[:,2] /= sc_w
        anchors[:,3] /= sc_h
        #print(i)
        #print(transformed_anchors.tolist())
        #print(anchors.tolist())
        anchor = anchors.tolist()[0]
        anchor = [int(np.round(num)) for num in anchor]
        iterm = name+','+str(anchor[0])+','+str(anchor[1])+','+str(anchor[2])+','+str(anchor[3])
        im = cv2.rectangle(im, (anchor[0], anchor[1]), (anchor[2], anchor[3]), (0,0,255), 3)
        #print(im)
        cv2.imwrite(os.path.join(parser.output_path, os.path.basename(i)), im)
        res.append(iterm)
    open('xpd.txt', 'w').write('\n'.join(res))

if __name__ == '__main__':
    main()


