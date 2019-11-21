import json
from pprint import pprint
import numpy as np
import shutil
from glob import glob 
import os
import cv2

# 计算准确和召回的函数
def calc_r_and_p(gt_path, pr_path, out_put_path=None, im_path='./3k_imgs/'):
    # gt_path: 每张图一个txt文件，里边有一行文本，以','为分隔符的tdlr
    # pr_path: 每张图一个txt文件，里边有一行文本，以','为分隔符的tdlr
    THRES = 20
    black_num = 0
    recall_num = 0
    error_num = 0
    predict_black_num = 0
    
    right_black_ims = []
    error_black_ims = []
    
    base_name_list = [os.path.basename(i) for i in glob(pr_path+'/*txt')]
    for base_name in base_name_list:
        g_tdlr = np.array(open(os.path.join(gt_path, base_name)).read().split(',')[:4]).astype(np.int16)
        p_tdlr = np.array(open(os.path.join(pr_path, base_name)).read().split(',')[:4]).astype(np.int16)
        if (g_tdlr > 20).sum() > 0:
            black_num += 1
            if (abs(p_tdlr - g_tdlr) > THRES).sum() > 0:
                error_num += 1
                error_black_ims.append(base_name)
            else:
#                 print(g_tdlr, p_tdlr)
                recall_num += 1
                right_black_ims.append(base_name)
        if (p_tdlr > 20).sum() > 0:
            predict_black_num += 1
    
    if out_put_path is not None:
        right_dir = os.path.join(out_put_path, 'right')
        error_dir = os.path.join(out_put_path, 'error')
        try:
            os.mkdir(right_dir)
        except:
            pass
        try:
            os.mkdir(error_dir)
        except:
            pass
        for txt_name in right_black_ims:
            im_name = txt_name.replace('.txt', '.jpeg')
            im_holl_name = os.path.join(im_path, im_name)
            save_name = os.path.join(right_dir, im_name)
            im = cv2.imread(im_holl_name)
            h,w,c = im.shape
            tdlr = np.array(open(os.path.join(pr_path, txt_name)).read().split(',')[:4]).astype(np.int16)
            x,y,xx,yy = tdlr_to_box(tdlr, w, h)
            im = cv2.rectangle(im, (x,y), (xx,yy), (0,0,255), 5)
            cv2.imwrite(save_name, im)
            demo_name = os.path.join(right_dir, 'cut_'+im_name)
            demo_im = im[y:yy,x:xx,:]
            cv2.imwrite(demo_name, demo_im)
        for txt_name in error_black_ims:
            im_name = txt_name.replace('.txt', '.jpeg')
            im_holl_name = os.path.join(im_path, im_name)
            save_name = os.path.join(error_dir, im_name)
            im = cv2.imread(im_holl_name)
            h,w,c = im.shape
            tdlr = np.array(open(os.path.join(pr_path, txt_name)).read().split(',')[:4]).astype(np.int16)
            x,y,xx,yy = tdlr_to_box(tdlr, w, h)
            im = cv2.rectangle(im, (x,y), (xx,yy), (0,0,255), 5)
            cv2.imwrite(save_name, im)
            demo_name = os.path.join(error_dir, 'cut_'+im_name)
            demo_im = im[y:yy,x:xx,:]
            cv2.imwrite(demo_name, demo_im)
        
        
    recall = recall_num/black_num
    precision = recall_num/predict_black_num
    lines = []
    lines.append("真实的黑边图片数量 %s, 召回 %s张, 错误 %s张" % (black_num, recall_num, error_num))
    lines.append("预测的黑边图片数量 %s, 正确 %s张, 错误 %s张" % (predict_black_num, recall_num, predict_black_num-recall_num))
    lines.append("黑边正确阈值: %d" % THRES)
    lines.append("准确率：%.02f%%" % (precision*100))
    lines.append("召回率：%.02f%%" % (recall*100))
    lines = '\n'.join(lines)
    print(lines)
    return lines




def tdlr_to_box(tdlr, w, h):
    t,d,l,r = [int(i) for i in tdlr]
    x = l
    xx = w - r
    y = t
    yy = h - d
    return (x,y,xx,yy)
    
def box_to_tdlr(box, w, h):
    x,y,xx,yy = [int(i) for i in box]
    t = y
    d = h - yy
    l = x
    r = w - xx
    return (t,d,l,r)