import numpy as np
from functools import partial
import pandas as pd
import os
from tqdm import tqdm_notebook, tnrange, tqdm
import sys
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
from torch import nn
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
from torch.optim import SGD,Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torch.optim.optimizer import Optimizer

import torchvision
from torchvision import models
import pretrainedmodels
from pretrainedmodels.models import *
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import transforms as T
from imgaug import augmenters as iaa
import random
import pathlib
import cv2

from torchvision import transforms
# from resnet import ResNet

from Transformer import Model as TransformerModel
from selfattention import SelfAttention
from selfattentiontovis import SelfAttention as SelfAttentiontovis
# from graph_attention import PyraformerModel

# from segment_anything import SamPredictor, sam_model_registry
# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
p=0.1
# torch.backends.cudnn.benchmark = False
# create dataset class
class MultiModalDataset(Dataset):
    def __init__(self,images_df, base_path,vis_path,augument=True,mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        if not isinstance(vis_path, pathlib.Path):
            vis_path = pathlib.Path(vis_path)
        self.images_df = images_df.copy() #csv
        self.augument = augument
        self.vis_path = vis_path #vist npy path
        self.images_df.Id = self.images_df.Id.apply(lambda x:base_path / str(x).zfill(6))
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        X = self.read_images(index)
        visit=self.read_npy(index).transpose(1,2,0)
        if not self.mode == "test":
            y = self.images_df.iloc[index].Target
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        X = T.Compose([T.ToPILImage(),T.ToTensor()])(X)
        visit=T.Compose([T.ToTensor()])(visit)
        return X.float(),visit.float(),y


    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        images = cv2.imread(filename+'.jpg')
        return images

    def read_npy(self,index):
        row = self.images_df.iloc[index]
        filename = os.path.basename(str(row.Id.absolute()))
        pth=os.path.join(self.vis_path.absolute(),filename+'.npy')
        visit=np.load(pth)
        return visit

    def augumentor(self,image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0,4),[
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            ], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
    
    def _reset(self, epoch, T_max):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        return CosineAnnealingLR(self.optimizer, self.T_max, self.eta_min, last_epoch=epoch)



class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
'''self-attention'''
# Decoupled spatial-temporal transformer for video inpainting (arXiv 2021)
import math
# import jittor as jt
# from jittor import nn


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, p=0.1):#, p=0.1
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                           ) / math.sqrt(query.size(-1))
        p_attn = nn.functional.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, tokensize, d_model, head, mode, p=0.1):
        super().__init__()
        self.mode = mode
        self.query_embedding = nn.Linear(d_model, d_model).to(device)
        self.value_embedding = nn.Linear(d_model, d_model).to(device)
        self.key_embedding = nn.Linear(d_model, d_model).to(device)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head
        self.h, self.w = tokensize
        
    # def execute(self, x, t):
    def forward(self, x, t):#([16, 2432, 4, 3])
       # print(x.size())
        #x=x.view(-1,7,1,26,24)#增加了一维度
        x=x.view(-1,2432,1,4,3)
        #x=x.view(-1,2432,1,1,1)
        x=x.permute(0,1,2,4,3)
       # print(x.shape)
        #x=x.contiguous().view(-1,1,7,24)#bt=bsize*T时相
        # x=x.view(-1,7,1,24)
        x=x.contiguous().view(-1,1,3,4)#bt=bsize*T时相#(-1,1,4,3)
       # print(x.size())
        bt, _,n ,c = x.size()#bt=bitchsize*26，26*32  n7 c24
        # bt,n ,_,c = x.size()#bt=bitchsize*26，26*32  n7 c24
        #print(x.size())
       # print(bt, n, c)
       # print(t)
        b = bt // t  #32/26，，，应该可以改为26*32/26=32，也可以不扩
        c_h = c // self.head
        # print(c_h)
        # device = torch.device('cuda:0')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        key = self.key_embedding(x).to(device)
        query = self.query_embedding(x).to(device)
        value = self.value_embedding(x).to(device)
        if self.mode == 's':
            key = key.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            query = query.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            value = value.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            att, _ = self.attention(query, key, value)
            #print(bt, n, c)
            #print(att.shape)
            att = att.permute(0, 1, 3, 2, 4).contiguous().view(bt, n, c)
        elif self.mode == 't':
            print(b)
            print(self.h,self.w)
            key = key.view(b, t, 2, self.h//2, 2, self.w//2, self.head, c_h)
            key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).view(
                b, 4, self.head, -1, c_h)
            query = query.view(b, t, 2, self.h//2, 2,
                               self.w//2, self.head, c_h)
            query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).view(
                b, 4, self.head, -1, c_h)
            value = value.view(b, t, 2, self.h//2, 2,
                               self.w//2, self.head, c_h)
            value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).view(
                b, 4, self.head, -1, c_h)
            att, _ = self.attention(query, key, value)
            att = att.view(b, 2, 2, self.head, t, self.h//2, self.w//2, c_h)
            att = att.permute(0, 4, 1, 5, 2, 6, 3,
                              7).contiguous().view(bt, n, c)
        output = self.output_linear(att).to(device)
        return output


class SpatialTemporalselfattention(nn.Module):
    def __init__(self):
        super().__init__()
         #自己加的
        self.linear = nn.Linear(2432, 64)
        self.dropout = nn.Dropout(p=p)
    def forward(self,x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # attention_block_s = MultiHeadedAttention(
        #     tokensize=[7, 24], d_model=24, head=4, mode='s').to(device)
        attention_block_s = MultiHeadedAttention(
            tokensize=[4, 3], d_model=4, head=4, mode='s').to(device)
        # tokensize=[4, 3], d_model=4, head=4, mode='s').to(device)
        # tokensize=[4, 8], d_model=64, head=4, mode='s')
        #attention_block_t = MultiHeadedAttention(
           # tokensize=[7, 24], d_model=24, head=4, mode='t').to(device)
        # input = torch.rand([8, 32, 64])
        input = x
        output = attention_block_s(input, 2432).to(device)
       # output = attention_block_t(output, 26).to(device)
        print(input.size(), output.size())#
        #自己增加的池化、全连接
        out=output.view(-1,2432,4,3)
        out = F.avg_pool2d(out, 3)#卷积核参数out = F.avg_pool2d((output.view(-1,2432,4,3)), 3)
        # print(out.shape)#([16, 2432, 1, 1])
        out = out.view(out.size(0), -1)#[batchsize=16, 2432]
        # print(out.shape)
        output = self.linear(out)
        output =self.dropout(output)
        # output=output.view(-1,29184)
        # =output.view(-1,26,7,24)
        print(output.shape)
        return output



    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))#layer2=([16, 192, 13, 12])
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))#([16, 192, 13, 12])
        x = self.shortcut(x)#x=([16, 304, 26, 24]) ,x=([16, 544, 13, 12])
        d = self.out_planes#512
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)#([16, 576, 13, 12])
        return out

def STAattention():
    return SpatialTemporalselfattention()





# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):  
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        # 使用自适应池化缩减map的大小，保持通道不变  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()
    
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)
        
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) 
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out



# Conv Transforme Block
class SelfAttentionConv(nn.Module):
    def __init__(self, k, headers = 8, kernel_size = 1, mask_next = True, mask_diag = False):
        super().__init__()
        
        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag
        
        h = headers
        
        # Query, Key and Value Transformations
        
        padding = (kernel_size-1)
        self.padding_opertor = nn.ConstantPad1d((padding,0), 0)
        
        self.toqueries = nn.Conv1d(k, k*h, kernel_size, padding=0 ,bias=True)
        self.tokeys = nn.Conv1d(k, k*h, kernel_size, padding=0 ,bias=True)
        self.tovalues = nn.Conv1d(k, k*h, kernel_size = 1 , padding=0 ,bias=False) # No convolution operated
        
        # Heads unifier
        self.unifyheads = nn.Linear(k*h, k)
    def forward(self, x):
        x=x.contiguous().view(-1,182,24)
        x=x.contiguous().permute(0,2,1)
        # x=([16, 2432, 4, 3])
        # x=x.contiguous().view(-1,64,12)
        # x=x.contiguous().permute(0,2,1)
        # Extraction dimensions
        b, t, k  = x.size() # batch_size, number_of_timesteps, number_of_time_series
        # print(b,t,k)
        # Checking Embedding dimension
        assert self.k == k, 'Number of time series '+str(k)+' didn t much the number of k '+str(self.k)+' in the initiaalization of the attention layer.'
        h = self.headers
        
        #  Transpose to see the different time series as different channels
        x = x.transpose(1,2)
        x_padded = self.padding_opertor(x)
        
        # Query, Key and Value Transformations
        queries = self.toqueries(x_padded).view(b,k,h,t)
        keys = self.tokeys(x_padded).view(b,k,h,t)
        values = self.tovalues(x).view(b,k,h,t)
        queries = x.view(b,k,h,t)
        keys = x.view(b,k,h,t)
        values = x.view(b,k,h,t)
        # Transposition to return the canonical format
        queries = queries.transpose(1,2) # batch, header, time serie, time step (b, h, k, t)
        queries = queries.transpose(2,3) # batch, header, time step, time serie (b, h, t, k)
        
        values = values.transpose(1,2) # batch, header, time serie, time step (b, h, k, t)
        values = values.transpose(2,3) # batch, header, time step, time serie (b, h, t, k)
        
        keys = keys.transpose(1,2) # batch, header, time serie, time step (b, h, k, t)
        keys = keys.transpose(2,3) # batch, header, time step, time serie (b, h, t, k)
        
        
        # Weights 
        queries = queries/(k**(.25))
        keys = keys/(k**(.25))
        
        queries = queries.transpose(1,2).contiguous().view(b*h, t, k)
        keys = keys.transpose(1,2).contiguous().view(b*h, t, k)
        values = values.transpose(1,2).contiguous().view(b*h, t, k)
        
        
        weights = torch.bmm(queries, keys.transpose(1,2))
        
                
        ## Mask the upper & diag of the attention matrix
        if self.mask_next :
            if self.mask_diag :
                indices = torch.triu_indices(t ,t , offset=0)
                weights[:, indices[0], indices[1]] = float('-inf')
            else :
                indices = torch.triu_indices(t ,t , offset=1)
                weights[:, indices[0], indices[1]] = float('-inf')
        
        # Softmax 
        weights = F.softmax(weights, dim=2)
        
        # Output
        output = torch.bmm(weights, values)
        output = output.view(b,h,t,k)
        output = output.transpose(1,2).contiguous().view(b,t, k*h)
        
        return self.unifyheads(output) # shape (b,t,k)
class ConvTransformerBLock(nn.Module):
    def __init__(self, k, headers, kernel_size =24, mask_next = True, mask_diag = False, dropout_proba = 0.2):
        super().__init__()

        self.attention = SelfAttentionConv(k, headers, kernel_size, mask_next, mask_diag)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.feedforward = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )
        self.dropout = nn.Dropout(p = dropout_proba)
        self.cls = nn.Linear(4368,64)
    def forward(self, x, train=False):
        x=x.view(-1,182,24)
        x=x.permute(0,2,1)
        # x=x.contiguous().view(-1,64,12)
        # x=x.contiguous().permute(0,2,1)
        # Self attention + Residual
        x = self.attention(x) + x
        
        # Dropout attention
        if train :
            x = self.dropout(x)
        
        # # First Normalization
        x = self.norm1(x)
    
        # Feed Froward network + residual
        x = self.feedforward(x) + x
        
        # Second Normalization
        x = self.norm2(x)

        # x=x.view(-1,4368)
        # # x=x.view(-1,768)
        # x = self.cls(x)
        x = self.dropout(x)
        # print(x.shape)
        return x
    

def ConvTransformer():
    return ConvTransformerBLock(182,1)#k, headers
    # return ConvTransformerBLock(64,1)#k, headers
conf={"task_name":'classification',
              "seq_len":182,"label_len":91,"moving_avg":25,
              "pred_len":182,"enc_in":24,"e_layers":2,"d_model":64,"embed":"","freq":"h","num_class":64,"top_k":1,"d_ff":512,"num_kernels":25,"dropout":0.5,"dec_in":24,"factor":1,"output_attention":0,"n_heads":1,"activation":1,"distil":1,"d_layers":1,"c_out":24,'embed_type':0}
 
# conf={"task_name":'classification',
#               "seq_len":182,"label_len":"","pred_len":24,"enc_in":182,"e_layers":1,"d_model":2,"embed":"","freq":"","num_class":64,"top_k":1,"d_ff":182,"num_kernels":24,"dropout":0.5,"dec_in":182,"factor":1,"output_attention":1,"n_heads":1,"activation":1,"distil":1,"d_layers":1,"c_out":182}
#    conf={"task_name":"classification",
#               "seq_len":24,"label_len":"","pred_len":24,"enc_in":24,"e_layers":2,"d_model":2,"embed":"","freq":"","num_class":64,"top_k":24,"d_ff":182,"num_kernels":3,"dropout":0.5}
    
# def ResNEt():
#     return ResNet()
#main
class MultiModalNet(nn.Module):
    def __init__(self, backbone1, backbone2, drop, pretrained=True):
        super().__init__()
        if pretrained:
            img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained='imagenet') #seresnext101
        else:
            img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained=None)
       
        #self.visit_model=DPN26()
        self.vis_img_attention=SelfAttention()
        self.img_vis_attention=SelfAttentiontovis()
        self.visitConv_model=ConvTransformer()

        self.conv_proj= models.vit_b_16(pretrained=True).conv_proj
        self.encoder= models.vit_b_16(pretrained=True).encoder
        self.heads= models.vit_b_16(pretrained=True).heads

        self.img_model16 = models.vit_b_16(pretrained=True)
        self.clss= nn.Linear(1000,256)
        self.sclss= nn.Linear(2100,256)
        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder)

        if drop > 0:
            self.img_fc = nn.Sequential(FCViewer(),
                                    nn.Dropout(drop),
                                    nn.Linear(img_model.last_linear.in_features, 256))
                                    
        else:
            self.img_fc = nn.Sequential(
                FCViewer(),
                nn.Linear(model.last_linear.in_features, 256)
            )
        
        self.toimgcls=nn.Linear(24,1)
        self.toviscls=nn.Linear(64,1)
        self.viscls = nn.Linear(768,64)
        self.visscls = nn.Linear(4368,64)
        self.viscls = nn.Linear(768,64)
        self.cls = nn.Linear(322,config.num_classes)
        self.dropout = nn.Dropout(drop) 
        self.dropout1 = nn.Dropout(drop) 
        self.img_cls = nn.Linear(256,config.num_classes)
        self.vis_cls = nn.Linear(64,config.num_classes)
        

    def forward(self, x_img,x_vis):
        resize =transforms.Resize([224,224])
        x_img = resize(x_img)

        x_img = self.img_model16(x_img)

        b, t = x_img.size()
        vit_output=x_img.view(-1,100,10)

        pooled_output = nn.MaxPool2d(kernel_size=(2, 2))(vit_output)
        x=pooled_output 

        conv_transpose1 = nn.ConvTranspose2d(in_channels=b, out_channels=b, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), output_padding=(0, 0)).to(device)
        conv_transpose2 = nn.ConvTranspose2d(in_channels=b, out_channels=b, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), output_padding=(0, 0)).to(device)
        conv_transpose1.weight = conv_transpose1.weight.to(device)
        conv_transpose2.weight = conv_transpose2.weight.to(device)
        x1 = conv_transpose1(x)
        x2 = conv_transpose2(x1)
       
        vit_output=vit_output.view(-1,1000)
        pooled_output=pooled_output.view(-1,250)
        x1=x1.view(-1,364)
        x2=x2.view(-1,486)
        x_img=torch.cat((vit_output,pooled_output,x1,x2),dim=1)
  
    
        x_img = self.sclss(x_img)

        x_img_tovis =x_img.view(-1,64,4)
        # x_img=x_img.view(-1,64)
        x_img=self.dropout(x_img)
        
        x_imgsoft=self.img_cls(x_img)
       
        x_vis=self.visitConv_model(x_vis)

        x_vis_toimg=self.vis_img_attention(x_vis)

        x_vis_toimg=x_vis_toimg.view(x_vis_toimg.size(0),-1)

        x_vis_toimg=self.toimgcls(x_vis_toimg)

        
        x_img=torch.cat((x_vis_toimg,x_img),1)


        x_vis = x_vis.view(x_vis.size(0), -1)
        x_vis=self.visscls(x_vis)

        x_vis=self.dropout(x_vis)
        x_vissoft=self.vis_cls(x_vis)
        x_softcat=torch.cat((x_imgsoft,x_vissoft),1)

        x_img_tovis=self.img_vis_attention(x_img_tovis)

        x_img_tovis=x_img_tovis.view(x_img_tovis.size(0),-1)

        x_img_tovis=self.toviscls(x_img_tovis)#64*1
  
        x_vis=torch.cat((x_img_tovis,x_vis),1)
  
        x_cat = torch.cat((x_img,x_vis),1)
        x_cat = self.cls(x_cat)
        x_cat=self.dropout(x_cat)
          # 双重融合
        x_catcat=torch.cat((x_softcat,x_cat),1)
        return x_cat,x_imgsoft,x_vissoft

        # return x_cat