from __future__ import print_function
from utils import *
import os 
import sys
from shutil import copyfile
import random 
import warnings
import time 
import json 
import numpy as np 
import pandas as pd 
from config import config
from notify import pushplus

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch 
import torchvision
from multimodal import MultiModalDataset,MultiModalNet,CosineAnnealingLR
from tqdm import tqdm 
from datetime import datetime
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import torch.nn.functional as F
from Focalloss import FocalLoss,Balanced_CE_loss,CombinedLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# use_dropout = True  # 不使用Dropout的情况下为False
# dropout_ratio = 0.2

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


log = Logger()
log.open(f"{config.logs}/{config.model_name}_log_train.txt",mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------|----------- Valid ---------|----------Best Results---|------------|\n')
log.write('mode     iter     epoch    |    acc  loss  f1_macro   |    acc  loss  f1_macro    |    loss  f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------|\n')


def train(train_loader,model,criterion,optimizer,epoch,valid_metrics,best_results,start):
    losses = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()
    #com = AverageMeter()
    com=[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
    b=np.array(com)
    model.train()
    for i,(images,visit,target) in enumerate(train_loader):
        visit=visit.to(device)
        images = images.to(device)
        indx_target=target.clone()
        target = torch.from_numpy(np.array(target)).long().to(device)
        #print(target)
        # compute output
        # output = model(images,visit)
        # #print(output)
        # loss = criterion(output,target)
        output = model(images,visit)[0]
        output1=model(images,visit)[1]
        output2=model(images,visit)[2]
        a=0.25
        loss = criterion(output,target)+a*(criterion(output1,target)+criterion(output2,target))
        
        losses.update(loss.item(),images.size(0))
        #print(target.cpu().data.numpy())
        f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
        acc_score=accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))
        com_c=confusion_matrix(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))
        a=np.array(com_c)
        if a.shape==b.shape:
          a=b+a
          b=a
        else:
          #print(a.shape)
          a=b
        # com.update(com_c,images.size(0))
        f1.update(f1_batch,images.size(0))
        acc.update(acc_score,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('\r',end='',flush=True)
        #print('\r')
        # print(com_c)
        message = '%s %5.1f %6.1f      |   %0.3f  %0.3f  %0.3f  | %0.3f  %0.3f  %0.4f   | %s  %s  %s |   %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                acc.avg, losses.avg, f1.avg,
                valid_metrics[0], valid_metrics[1],valid_metrics[2],
                str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    print(b)
    #log.write(message)
    #log.write("\n")
    return [acc.avg,losses.avg,f1.avg,com]

# 2. evaluate function
def evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start):
    # only meter loss and f1 score
    losses = AverageMeter()
    f1 = AverageMeter()
    acc= AverageMeter()
    com=[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
    b=np.array(com)
    # switch mode for evaluation
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (images,visit,target) in enumerate(val_loader):
            images_var = images.to(device)
            visit=visit.to(device)
            indx_target=target.clone()
            target = torch.from_numpy(np.array(target)).long().to(device)
            # output = model(images_var,visit)
            # loss = criterion(output,target)
            output = model(images_var,visit)[0]
            output1=model(images_var,visit)[1]
            output2=model(images_var,visit)[2]
            a=0.25
            loss = criterion(output,target)+a*(criterion(output1,target)+criterion(output2,target))
             
            losses.update(loss.item(),images_var.size(0))
            f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
            acc_score=accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))        
            com_c=confusion_matrix(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))
            a=np.array(com_c)
            if a.shape==b.shape:
              a=b+a
              b=a
             # print(b)
            else:
             # print(a.shape)
              a=b
            f1.update(f1_batch,images.size(0))
            acc.update(acc_score,images.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f     |     %0.3f  %0.3f   %0.3f    | %0.3f  %0.3f  %0.4f  | %s  %s  %s  |  %s' % (\
                    "val", i/len(val_loader) + epoch, epoch,                    
                    acc.avg,losses.avg,f1.avg,
                    train_metrics[0], train_metrics[1],train_metrics[2],
                    str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                    time_to_str((timer() - start),'min'))

            print(message, end='',flush=True)
        log.write("\n")
        print(b)
        #log.write(message)
        #log.write("\n")
        
    return [acc.avg,losses.avg,f1.avg]

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model,folds):
    sample_submission_df = pd.read_csv("./test.csv")
    #3.1 confirm the model converted to cuda
    filenames,labels ,submissions= [],[],[]
    model.to(device)
    model.eval()
    submit_results = []
    for i,(input,visit,filepath) in tqdm(enumerate(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.to(device)
            visit=visit.to(device)
            y_pred = model(image_var,visit)
            #print(y_pred)
            label=F.softmax(y_pred).cpu().data.numpy()
            #print(label)
            labels.append(label==np.max(label))
            filenames.append(filepath)
            com_c=confusion_matrix(y_pred.cpu().data.numpy(),np.argmax(F.softmax(y_pred).cpu().data.numpy(),axis=1))
            #com.update(com_c,images.size(0))
            print('\r')
            print(com_c)
    for row in np.concatenate(labels):
        subrow=np.argmax(row)
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/%s_bestloss_submission.csv'%config.model_name, index=None)


# 4. main function
def main():
    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    
    #4.2 get model
    model=MultiModalNet("se_resnext101_32x4d","dpn26",0.5)
    #model=MultiModalNet("se_resnet50","dpn26",0.5)


    #L1+L2+cross
    # def l1_regularization(model, l1_alpha):
    #     l1_loss = []
    #     for module in model.modules():
    #         if type(module) is nn.BatchNorm2d:
    #             l1_loss.append(torch.abs(module.weight).sum())
    #     return l1_alpha * sum(l1_loss)
 
    # def l2_regularization(model, l2_alpha):
    #     l2_loss = []
    #     for module in model.modules():
    #         if type(module) is nn.Conv2d:
    #             l2_loss.append((module.weight ** 2).sum() / 2.0)
    #     return l2_alpha * sum(l2_loss)


    #4.3 optim & criterion
    # 原来
    optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(),lr=config.lr,weight_decay=0.01)
    # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=0.018)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5.0)
#torch.optim.Adam()参数中的 weight_decay=5.0 即为L2正则化（只是pytorch换了名字），其数值即为L2正则化的惩罚系数，一般设置为1、5、10（根据需要设置，默认为0，不使用L2正则化）
    criterion=nn.CrossEntropyLoss().to(device)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # print(c)
    # criterion=CombinedLoss()
    # FocalLoss().to(device)
    # print(criterion)
    # Balanced_CE_loss
    # nn.CrossEntropyLoss().to(device)+
    # criterion=nn.CrossEntropyLoss().to(device)+l1_regularization(model,0.1)+l2_regularization(model,0.1)


    

    start_epoch = 0
    best_acc=0
    best_loss = np.inf
    best_f1 = 0
    best_results = [0,np.inf,0]
    val_metrics = [0,np.inf,0]
    resume = False
    if resume:
        checkpoint = torch.load('/data/ruiyang.sun/project/vitae-b-checkpoint-1599-transform-no-average.pth')
        # checkpoint = torch.load('./checkpoints/best_models/seresnext101_dpn92_defrog_multimodal_fold_0_model_best_loss.pth.tar')
        best_acc = checkpoint['best_acc']
        best_loss = checkpoint['best_loss']
        best_f1 = checkpoint['best_f1']
        start_epoch = checkpoint['epoch']

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    all_files = pd.read_csv("./train.csv")
    test_files = pd.read_csv("./test.csv")
    train_data_list,val_data_list = train_test_split(all_files, test_size=0.1, random_state = 2050)
    
    # load dataset
    train_gen = MultiModalDataset(train_data_list,config.train_data,config.train_vis,mode="train")
    train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=1) #num_worker is limited by shared memory in Docker!

    val_gen = MultiModalDataset(val_data_list,config.train_data,config.train_vis,augument=False,mode="train")
    val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=1)

    test_gen = MultiModalDataset(test_files,config.test_data,config.test_vis,augument=False,mode="test")
    test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=1)

    # scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    #n_batches = int(len(train_loader.dataset) // train_loader.batch_size)
    #scheduler = CosineAnnealingLR(optimizer, T_max=n_batches*2)
    start = timer()

    #train
    reslist = []
    for epoch in range(0,config.epochs):
        scheduler.step(epoch)
        # train
        train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start)
        # val
        val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start)
        # check results
        is_best_acc=val_metrics[0] > best_results[0] 
        best_results[0] = max(val_metrics[0],best_results[0])
        is_best_loss = val_metrics[1] < best_results[1]
        best_results[1] = min(val_metrics[1],best_results[1])
        is_best_f1 = val_metrics[2] > best_results[2]
        best_results[2] = max(val_metrics[2],best_results[2])   
        # save model
        save_checkpoint(model,
            {
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_acc":best_results[0],
                    "best_loss":best_results[1],
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "best_f1":best_results[2],
        },is_best_acc,is_best_loss,is_best_f1,fold)
        # print logs
        print('\r',end='',flush=True)
        log.write('%s  %5.1f %6.1f      |   %0.3f   %0.3f   %0.3f     |  %0.3f   %0.3f    %0.3f    |   %s  %s  %s | %s' % (\
                "best", epoch, epoch,                    
                train_metrics[0], train_metrics[1],train_metrics[2],
                val_metrics[0],val_metrics[1],val_metrics[2],
                str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                time_to_str((timer() - start),'min'))
            )
        log.write("\n")
        reslist.append({"epoch":epoch,"acc":str(best_results[0])[:5],"f1":str(best_results[2])[:5]})
        time.sleep(0.01)
        if (epoch+1)%5==0 or epoch==0:
            pushplus(config.model_name,pd.DataFrame(reslist).to_markdown(index=False))

    best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
    model.load_state_dict(best_model["state_dict"])
    test(test_loader,model,fold)


if __name__ == "__main__":
    main()
