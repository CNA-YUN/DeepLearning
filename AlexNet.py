import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        '''
        这里搭建卷积层，需要按顺序定义卷积层、
        激活函数、最大池化层、卷积层、激活函数、
        最大池化层、卷积层、激活函数、卷积层、
        激活函数、卷积层、激活函数、最大池化层，
        具体形状见测试说明
        '''
        self.conv = nn.Sequential(
            ########## Begin ##########
            nn.Conv2d(3,96,11,stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
            ########## End ##########
        )
        '''
        这里搭建全连接层，需要按顺序定义
        全连接层、激活函数、丢弃法、
        全连接层、激活函数、丢弃法、全连接层，
        具体形状见测试说明
        '''
        self.fc = nn.Sequential(
            ########## Begin ##########
            nn.Linear(6400,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5,inplace=False),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,1000)
            ########## End ##########
        )

    def forward(self, img):
        '''
        这里需要定义前向计算
        '''
        ########## Begin ##########
        img=self.conv(img)
        img=self.fc(img)
        return img
        ########## End ##########

model = AlexNet()
print(model)