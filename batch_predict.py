import os
from dataloader import Resizer, Normalizer
from torchvision import datasets, models, transforms
import cv2
import numpy as np
from glob import glob
import torch
import time
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from recall_tool import calc_r_and_p

#定义一个数据集
class PredictDataset(Dataset):
    """ 数据集演示 """
    def __init__(self, im_ll):
        """ 保存图片列表，初始化图片预处理器
        """
        self.imll = im_ll
        self.transform=transforms.Compose([Normalizer(), Resizer()])

    def __len__(self):
        """ 返回图片列表长度
        """
        return len(self.imll)
    def _train_input(self, index):
        annot = np.array([[10,10,20,20,0],], dtype=np.float64)
        im = cv2.imread(self.imll[index])
        h,w,c = im.shape
        img = im.astype(np.float32)/255.0
        samp = {'img':img, 'annot':annot}
        model_input = self.transform(samp)
        img = model_input['img'].float().permute((2,0,1)).contiguous()
        sc_h = model_input['sc_h']
        sc_w = model_input['sc_w']
        return img, sc_w, sc_h, w, h
    def __getitem__(self, idx):
        '''
        根据 idx 返回一行数据
        '''
        return self._train_input(idx)


def predict(im_ll, model_path='/home/hao.wyh/code/git/pytorch-retinanet/output_models/true_data_v2_mix/coco_retinanet_3.pt'):
    retinanet = torch.load(model_path)
    retinanet = retinanet.cuda()
    retinanet.eval()
    B = PredictDataset(im_ll)
    L = DataLoader(B, batch_size=1, shuffle=False, num_workers=10)
    res_list = []
    w_h_list = []
    print(len(B))
    print(time.time())
    for x, sc_w_list, sc_h_list, w_list, h_list in tqdm(L):
        sc_w_list = sc_w_list.tolist()
        sc_h_list = sc_h_list.tolist()
        w_list = w_list.tolist()
        h_list = h_list.tolist()
        scores, classification, transformed_anchors = retinanet(x.cuda())
        anchors = transformed_anchors.cpu().tolist()
        length = len(sc_w_list)

        for i in range(length):
            anchor = anchors[i]
            anchor[0] /= sc_w_list[i]
            anchor[1] /= sc_h_list[i]
            anchor[2] /= sc_w_list[i]
            anchor[3] /= sc_h_list[i]
            res_list.append(anchor[:4])
            w_h_list.append((w_list[i], h_list[i]))
    print(time.time())
    return res_list, w_h_list

def test(pr_path):
        gt_path = '/home/hao.wyh/jupyter/黑边/评估任务/3k_imgs/'
        im_path = gt_path
        out_content = calc_r_and_p(gt_path=gt_path, pr_path=pr_path, out_put_path=pr_path, im_path=im_path)
        # if save_res_info_path is not None:
        open(os.path.join(pr_path, 'res.res'), 'w').write(out_content)

def main():
    transform=transforms.Compose([Normalizer(), Resizer()])
    annot = np.array([[10,10,20,20,0],], dtype=np.float64)

    parser = argparse.ArgumentParser(description='测试模型效果.')
    #parser.add_argument('--model', help='Path to model (.pt) file.', default='/home/hao.wyh/code/git/pytorch-retinanet/output_models/main_detect_v11_restart/model_final.pt')
    parser.add_argument('-m', dest='model', help='Path to model (.pt) file.', default='/home/hao.wyh/code/git/pytorch-retinanet/output_models/true_data_v2_mix/coco_retinanet_3.pt')
    #parser.add_argument('--model', help='Path to model (.pt) file.', default='output_models/main_detect_v10_deeper8/model_final.pt')
    parser.add_argument('-o', dest='output_path', help='Path to save output imgs.', default='./tmp_out/')
    #parser.add_argument('--input_path', help='Path to save output imgs.', default='/home/hao.wyh/jupyter/黑边/smart_reverse_label')
    #parser.add_argument('--input_path', help='Path to save output imgs.', default='/home/hao.wyh/jupyter/黑边/评估任务/black_imgs')
    parser.add_argument('-i', dest='input_path', help='Path to save output imgs.', default='/home/hao.wyh/jupyter/黑边/评估任务/3k_imgs')
    parser.add_argument('-s', dest='show_out_im', action="store_true" , help='是否测试模型准召率')
    parser.add_argument('-t', dest='test', action="store_true" , help='是否测试模型准召率')
    parser.add_argument('-ot', dest='only_test', action="store_true" , help='是否测试模型准召率')
    parser = parser.parse_args()

    if parser.only_test:
        test(parser.output_path)
        exit()

    ll = glob(parser.input_path+'/*jpg')
    if len(ll) == 0:
        ll = glob(parser.input_path+'/*jpeg')
    if not os.path.exists(parser.output_path):
        os.mkdir(parser.output_path)
    res_list, w_h_list = predict(ll, parser.model)
    res = []
    for idx in tqdm(range(len(res_list))):
        i = ll[idx]
        name = i.split('/')[-1]
        anchor = res_list[idx]
        anchor = [int(np.round(num)) for num in anchor]
        iterm = name+','+str(anchor[0])+','+str(anchor[1])+','+str(anchor[2])+','+str(anchor[3])
        res.append(iterm)
        if parser.show_out_im:
            im = cv2.imread(i)
            im = cv2.rectangle(im, (anchor[0], anchor[1]), (anchor[2], anchor[3]), (0,0,255), 3)
            cv2.imwrite(os.path.join(parser.output_path, os.path.basename(i)), im)
        # for i in open('./xpd.txt').read().split('\n'):
        name,x,y,xx,yy = iterm.split(',')
        x,y,xx,yy = [int(i) for i in [x,y,xx,yy]]
        w, h = w_h_list[idx]
        # print(x,y,xx,yy,w,h, name)
        t = y
        d = h - yy
        l = x
        r = w - xx
        t,d,l,r = [str(i) for i in [t,d,l,r]]
        open(os.path.join(parser.output_path, name.replace('.jpeg', '.txt')), 'w').write(','.join([t,d,l,r]))
    open(os.path.join(parser.output_path,'xpd.res'), 'w').write('\n'.join(res))

    if parser.test:
        test(parser.output_path)


if __name__ == '__main__':
    main()


