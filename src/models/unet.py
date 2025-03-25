# Implement your UNet model here

import warnings
warnings.filterwarnings("ignore")
import torch 
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, stride=1,kernel_size=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride = stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            

        )

    def forward(self, x):
        return self.triple_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__() #??????????????????nn.Module???????????????????????????????????????????????????????????????????????????
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, 2,stride=2)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2,stride=2)
        self.up_conv3 = nn.ConvTranspose2d(256,128, 2,stride=2)
        self.up_conv4 = nn.ConvTranspose2d(128,64, 2,stride=2)
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128,stride=1)

        self.down2 = DoubleConv(128, 256,stride=1)

        self.down3 = DoubleConv(256, 512,stride=1)
        self.down4 = DoubleConv(512, 1024,stride=1)

        self.up1 = DoubleConv(1024, 512)
        self.up2 = DoubleConv(512, 256)
        self.up3 = DoubleConv(256,128)
        self.up4 = DoubleConv(128,64)

        # self.outfusion = DoubleConv(1, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(0.3)
    

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
       
        # print(x1.shape)
        x2 = self.pooling(x1)
        x2 = self.down1(x2)
        
        # print(x2.shape)
        x3 = self.pooling(x2)
        x3 = self.down2(x3)
        
        # print(x3.shape)
        x4 = self.pooling(x3)
        x4 = self.down3(x4)
        
        # print(x4.shape)
        x5 = self.pooling(x4)
        x5 = self.down4(x5)
        # print(x5.shape)
        

        x5 = self.up_conv1(x5)#28X28=>56X56
        x5 = self.drop(x5)
        # print(x5.shape)
        x=torch.cat([x5, x4], dim=1)
        # print(x.shape)
        x = self.up1(x) #conv=>
        # print(x.shape)
        
                
        x = self.up_conv2(x)
        x = self.drop(x)
        # print(x.shape)
        x=torch.cat([x, x3], dim=1)
        # print(x.shape)
        x = self.up2(x)
       
        # print(x.shape)


        x = self.up_conv3(x)
        x = self.drop(x)
        # print(x.shape)
        x=torch.cat([x, x2], dim=1)
        # print(x.shape)
        x = self.up3(x)
        # print(x.shape)

        x = self.up_conv4(x)
        x = self.drop(x)
        # print(x.shape)
        x=torch.cat([x, x1], dim=1)
        # print(x.shape)
        x = self.up4(x)
        # print(x.shape)
  
        logits = self.outc(x)
        # print(logits.shape)
        #logits = torch.softmax(logits,dim=1)
        # logits = torch.argmax(logits,dim=1)
        return logits
    

# https://github.com/TommyHuang821/Pytorch_DL_Implement/blob/main/10_pytorch_SemanticSegmentation_VOC2007.ipynb
