import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from model.layers import *
        

class TestUNet(nn.Module):

    def __init__(self,in_channels,n_class):
        super().__init__()
                
        self.dconv_down1 = Residual(in_channels, 64,use_1x1conv=True)
        self.dconv_down2 = Residual(64, 128,use_1x1conv=True)
        self.dconv_down3 = Residual(128, 256,use_1x1conv=True)
        self.dconv_down4 = Residual(256, 512,use_1x1conv=True)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = Residual(512, 256,use_1x1conv=True)
        self.dconv_up2 = Residual(256 + 256, 128,use_1x1conv=True)
        self.dconv_up1 = Residual(256, 64,use_1x1conv=True)
        
        self.conv_last = nn.Conv2d(128, n_class, 1)
        
        
    def forward(self, x):
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)
        x = self.maxpool(x)
        x = self.upsample(x)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x,conv3],dim=1)
        
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x,conv2],dim=1)
        
        x = self.dconv_up1(x)
        x = self.upsample(x)
        x = torch.cat([x,conv1],dim=1)
        
        out = self.conv_last(x)

        return out



class UNet(nn.Module):
    
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.down1 = down(input_channels, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 256)

        self.up1 = up(256, 128, output_pad=1, concat=False)
        self.up2 = up(128+128, 64)
        self.up3 = up(64+64, 32)
        self.up4 = up(32+32, output_channels, final=True, tanh=False)

    # Adjusting for the input of real data, 176x176
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, None)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x


