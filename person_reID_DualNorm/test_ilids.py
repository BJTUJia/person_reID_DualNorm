# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import models


######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--which_epoch',default='129', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../test/iLIDS',type=str, help='./test_data')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')


opt = parser.parse_args()


str_ids = opt.gpu_ids.split(',')
test_dir = opt.test_dir

torch.manual_seed(opt.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Currently using GPU {}".format(opt.gpu_ids))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(opt.seed)

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



data_dir = test_dir
image_datasets = datasets.ImageFolder(data_dir,data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                              shuffle=False, num_workers=4)

use_gpu = torch.cuda.is_available()


def load_network(network):
    save_path = os.path.join('../model/net_%s.pth'%opt.which_epoch)
    checkpoint = torch.load(save_path)

    # for k, v in checkpoint.items():
    #     print(k)
    network.load_state_dict(checkpoint)
    return network

#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W N:TensornSamples in minibatch, i.e., batchsize x nChannels x Height x Width
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0 
    for data in dataloaders:
        img,_= data
        n, c, h, w = img.size()
        count += n
        print(count)
        
        ff = torch.FloatTensor(n,1280).zero_()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            #print(f.size())
            ff = ff+f

        
        
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features


######################################################################
# Load Collected data Trained model
print('-------test-----------')

model_structure = models.init_model(name='mobilenet', num_classes=17685, training=False, use_gpu=use_gpu)
# print(model_structure)
model_structure = model_structure.cuda()
model = load_network(model_structure)


# Change to test mode
model = model.eval()

# Extract feature

feature= extract_feature(model,dataloaders)


# Save to Matlab for check
result = {'feature':feature.numpy()}
scipy.io.savemat('../data/iLIDS/pytorch_result_ilds_%s.mat'%opt.which_epoch,result)

