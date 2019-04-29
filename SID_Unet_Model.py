import torch
import torch.nn as nn
import torch.nn.functional as F



class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    An activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        
        self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, kernel_size = 3, stride=1, padding=1)
    
        self.conv2 = nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = 3,
                               stride=1, padding=1)
        
        self.activ = nn.LeakyReLU(0.2)
    

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        
        x = self.activ(self.conv1(x))
        x = self.activ(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool
    

class UpConv(nn.Module):
    """
    A helper Module that performs 1 Transpose convolution and 2 convolutions
    The output of the transpose convolution is concatenated with 
    the output of a DownConv (before max pool)
    An activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels):    #512, 256
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.convT = nn.ConvTranspose2d(self.in_channels,self.out_channels,kernel_size=2,stride=2)
        
        self.conv1 = nn.Conv2d(in_channels = 2 * self.out_channels, out_channels = self.out_channels, kernel_size = 3, stride=1, padding=1)
    
        self.conv2 = nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = 3, 
                               stride=1, padding=1)
        
        self.activ = nn.LeakyReLU(0.2)
    
        
    def forward(self, x1, x2):
        x1 = self.convT(x1)
        x = x = torch.cat((x1,x2), dim =1)  #dim = 1 =>channels
        
        x = self.activ(self.conv1(x))
        x = self.activ(self.conv2(x))
        
        return x
        


class Unet(nn.Module):
    def __init__(self,  channels1 = 32):
        
        super().__init__()
        
        self.downconv1 =  DownConv(4,channels1)
        self.downconv2 =  DownConv(channels1,2 * channels1)
        self.downconv3 =  DownConv(2 * channels1, 4 * channels1)
        self.downconv4 =  DownConv(4 * channels1,8 * channels1)
        self.downconv5 =  DownConv(8 * channels1,16 * channels1, pooling=False)
        
        self.upconv1 = UpConv(16 * channels1, 8 * channels1)
        self.upconv2 = UpConv(8 * channels1, 4 * channels1)
        self.upconv3 = UpConv(4 * channels1, 2 * channels1)
        self.upconv4 = UpConv(2 * channels1, channels1)
        
        self.lastconv = nn.Conv2d(in_channels = channels1, out_channels = 12, kernel_size = 1, stride=1, padding=0)
        
        self.upscale = torch.nn.PixelShuffle(2)    # is it same as tf.depth_to_space?
        
        
    def forward(self, x):    # input of shape (N, C, H, W)
            
        
        
        
        x, conv1 = self.downconv1(x)
        x, conv2 = self.downconv2(x)
        x, conv3 = self.downconv3(x)
        x, conv4 = self.downconv4(x)
        x, _ = self.downconv5(x)
        
        x = self.upconv1(x,conv4)
        x = self.upconv2(x,conv3)
        x = self.upconv3(x,conv2)
        x = self.upconv4(x,conv1)
        
        x = self.lastconv(x)    #no activation
        
        x = self.upscale(x)
        
        #x = torch.sigmoid(x)
        
        
       
        
        return x
        
        
            