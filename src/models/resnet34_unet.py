# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1,kernel_size=3,shortcut=None ):
        super(Encoder,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
  
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride =1, padding=1),# stride=1,padding=1??????????????????
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5),
        )
        self.right = shortcut
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        # print("residual shape:"+ str(residual.shape))
        # print("out_shape: "+str(out.shape))
        out = out + residual
        out  = F.relu(out)
        return out


class Decoder(nn.Module):
    """(upsample => convolution => ReLU => [BN] => cbam ) * 2"""
    def __init__(self, in_channels,out_channels,stride=1,kernel_size=3,mid_channels=None ):
        super(Decoder,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=kernel_size,stride=2,padding=1,output_padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5),
        )
    def forward(self, x):
        # assert "not implement it yet"
        return self.conv(x)


class ResNet34_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNet34_UNet, self).__init__() #??????????????????nn.Module???????????????????????????????????????????????????????????????????????????
        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.inc = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2,2,1) 

        )
        self.down1 = self.make_Encoder_layer(64,64,3)
        self.down2 = self.make_Encoder_layer(64, 128,4, stride=2)
        self.down3 = self.make_Encoder_layer(128, 256,6, stride=2)
        self.down4 = self.make_Encoder_layer(256, 512,3,stride=2)
        self.down5 = self.make_Encoder_layer(512, 1024,3,stride=2)

        self.up1 = nn.ConvTranspose2d(1024, 512,2,stride=2,padding=0) 
        self.up2 = Decoder(1024,256)
        self.up3 = Decoder(512,128)
        self.up4 = Decoder(256,64)
        self.up5 = Decoder(128,64)
        self.up6 = Decoder(64,64)

        self.out =nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

        # self.logic = self.outc()
        
    

    def make_Encoder_layer(self, in_channel, out_channel, block_num, stride=1):
        # shortcut?????????????????????block??????????????????????????????????????????1d conv???????????? 
        # ???????????????????????????shape(stride=2)?????????????????? 
        shortcut = nn.Sequential( 
            nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False), 
            nn.BatchNorm2d(out_channel), 
        ) 
        layers = [] # ????????????ResidualBlock???????????????shape(??????stride)??????????????????????????? 
        layers.append(Encoder(in_channel, out_channel, stride=stride, shortcut=shortcut)) #????????????????????????????????????ResidualBlock????????????????????????????????????????????????????????????shortcut???????????? 
        for i in range(1, block_num): 
            layers.append(Encoder(out_channel, out_channel)) 

        return nn.Sequential(*layers)

    def outc(self, x):
        out = self.out(x)
        logic = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
        return logic

    def forward(self,x):

        # print(x.shape)
        x1 = self.inc(x)#t
        # print(x1.shape)#

        # x2 = self.pooling(x1)
        # print(x2.shape)#

        x3 = self.down1(x1)
        # print(x3.shape)#

        x4 = self.down2(x3)
        # print(x4.shape)#

        x5 = self.down3(x4)
        # print(x5.shape)#

        x6 = self.down4(x5)
        # print(x6.shape)#

        x7 = self.down5(x6)
        # print(x7.shape) #

        x= self.up1(x7)#
        # print(x.shape) 
        x= torch.cat([x6,x],dim=1)
        # print(x.shape)#

        x = self.up2(x)
        # print(x.shape)#

        x= torch.cat([x5,x],dim=1)
        # print(x.shape)#

        x = self.up3(x)
        # print(x.shape)#
        x= torch.cat([x4,x],dim=1)
        # print(x.shape)#

        x = self.up4(x)
        # print(x.shape)#
        x = torch.cat([x3,x],dim=1)
        # print(x.shape)#

        x = self.up5(x)
        # print(x.shape)#

        x = self.up6(x)#
        # print(x.shape)

        x = self.outc(x)#
        # print(x.shape)
        



        return x





    
