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
from torchvision import datasets,  transforms
from torchvision import models as models

import time
import os
import models
from losses import CrossEntropyLabelSmooth

# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_dir',default='../all',type=str, help='training dir path')
parser.add_argument('--train_all', default=True,action='store_true', help='use all training data' )
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--resume',default='',type=str,help='PATH')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')


opt = parser.parse_args()
data_dir = opt.data_dir

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
transform_train_list = [
            transforms.Resize((256, 128), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

print(transform_train_list)


data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                n, c, h, w = inputs.size()
                if n < opt.batchsize:  # skip the last batch
                    continue
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                # running_loss += loss.data[0]
                running_loss += loss.item()
                running_corrects += float(torch.sum(preds == labels.data))



            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch%10 == 9:
                    save_network(model, epoch)
                #draw_curve(epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model




######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('../model',save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()

def load_state_dict(network, state_dict=''):
    if state_dict:
        checkpoint=torch.load(state_dict)
        network.load_state_dict(checkpoint)
        return network

model = models.init_model(name='mobilenet_ifn', num_classes=len(class_names))

print(model)
model=model.cuda()

state_dict=opt.resume
load_state_dict(model,state_dict)

criterion = CrossEntropyLabelSmooth(num_classes=len(class_names), use_gpu=use_gpu)
# criterion = nn.CrossEntropyLoss()

# for mobilenetv2
optimizer_ft = optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4, nesterov=True)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

# for resnet50
# ignored_params = list(map(id, model.fc.parameters() ))
# base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
# optimizer_ft = optim.SGD([
#              {'params': base_params, 'lr': 0.1*0.05},
#              {'params': model.fc.parameters(), 'lr': 0.05}
# ], weight_decay=5e-4, momentum=0.9, nesterov=True)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=150)


