# -*- coding: utf-8 -*-
import argparse
import os
import torch
import copy
import random
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms
from PIL import Image
from flyai.data_helper import DataHelper
from flyai.dataset import Dataset
from flyai.framework import FlyAI

import numpy as np

from path import MODEL_PATH
from path import DATA_PATH

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

epochs=args.EPOCHS, batch=args.BATCH

class FlyAIDataset(Dataset):
  def __init__(self, x_dict, y_dict, train_flag=True):
      self.images = [x['image_path'] for x in x_dict]
      self.labels = [y['label'] for y in y_dict]
      if train_flag:
          self.transform = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.RandomHorizontalFlip(), # 随机水平翻转
                  transforms.RandomVerticalFlip(), # 随机竖直翻转
                  transforms.RandomRotation(30), #（-30，+30）之间随机旋转
                  transforms.ToTensor(), #转成tensor[0, 255] -> [0.0,1.0]
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
      else:
          self.transform = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  def __len__(self):
      return len(self.images)

  def __getitem__(self, index):
      path = os.path.join(DATA_PATH, self.images[index])
      image = Image.open(path)
      img = self.transform(image)
      label = self.labels[index]
      return img, label

data = Dataset()
x_train, y_train, x_val, y_val = data.get_all_data() # 获取全量数据
# x_train: [{'image_path': 'img/10479.jpg'}, {'image_path': 'img/14607.jpg'},   {'image_path': 'img/851.jpg'}...]
# y_train: [{'label': 39}, {'label': 4}, {'label': 3}...]
train_dataset = FlyAIDataset(x_train, y_train)
val_dataset = FlyAIDataset(x_val, y_val, train_flag=False)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.BATCH)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.BATCH)

#CNN Network


class ConvNet(nn.Module):
    def __init__(self,num_classes=6):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
        
        
        
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
            #Above output will be in matrix form, with shape (256,32,75,75)
            
        output=output.view(-1,32*75*75)
            
            
        output=self.fc(output)
            
        return output
            
      

class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("Food2")

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        pass

    def train(self):

        model=ConvNet(num_classes=6).to(device)

        #Optmizer and loss function
        optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
        loss_function=nn.CrossEntropyLoss()

        #calculating the size of training and testing images
        train_count=len(train_dataset)
        val_count=len(val_dataset)

        #Model training and saving best model
        best_accuracy=0.0

        for epoch in range(epochs):
            #Evaluation and training on training dataset
            model.train()
            train_accuracy=0.0
            train_loss=0.0
    
            for i, (images,labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images=Variable(images.cuda())
                    labels=Variable(labels.cuda())
            
                optimizer.zero_grad()
        
                outputs=model(images)
                loss=loss_function(outputs,labels)
                loss.backward()
                optimizer.step()
        
        
            train_loss+= loss.cpu().data*images.size(0)
            _,prediction=torch.max(outputs.data,1)
        
            train_accuracy+=int(torch.sum(prediction==labels.data))
        
        train_accuracy=train_accuracy/train_count
        train_loss=train_loss/train_count
    
    
        # Evaluation on testing dataset
        model.eval()
    
        test_accuracy=0.0
        for i, (images,labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images=Variable(images.cuda())
                labels=Variable(labels.cuda())
            
            outputs=model(images)
            _,prediction=torch.max(outputs.data,1)
            test_accuracy+=int(torch.sum(prediction==labels.data))
    
        test_accuracy=test_accuracy/val_count
    
    
        print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
    
        #Save the best model
        if test_accuracy>best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy=test_accuracy



if __name__ == '__main__':
    main = Main()
    print("下载数据")
    main.download_data()
    main.train()


class FlyAIDataset(Dataset):
  def __init__(self, x_dict, y_dict, train_flag=True):
      self.images = [x['image_path'] for x in x_dict]
      self.labels = [y['label'] for y in y_dict]
      if train_flag:
          self.transform = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.RandomHorizontalFlip(), # 随机水平翻转
                  transforms.RandomVerticalFlip(), # 随机竖直翻转
                  transforms.RandomRotation(30), #（-30，+30）之间随机旋转
                  transforms.ToTensor(), #转成tensor[0, 255] -> [0.0,1.0]
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
      else:
          self.transform = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  def __len__(self):
      return len(self.images)

  def __getitem__(self, index):
      path = os.path.join(DATA_PATH, self.images[index])
      image = Image.open(path)
      img = self.transform(image)
      label = self.labels[index]
      return img, label

