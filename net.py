import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
VGG-flavored UNet
'''
class UNet(nn.Module):
    def __init__(self,d_out):
        super().__init__()

        def conv_block(n_in,n_out,stride=1):
            #return nn.Sequential(nn.Conv2d(n_in,n_out,kernel_size=3,padding=1),nn.ReLU(inplace=True))
            return nn.Sequential(nn.Conv2d(n_in,n_out,kernel_size=3,stride=stride,padding=1),nn.BatchNorm2d(n_out),nn.ReLU(inplace=True))
        #

        self.down_conv = nn.ModuleList()
        block_channels = [16,32,64,128,256,512,512]
        n_blocks = len(block_channels)
        in_channels = 1
        self.is_down_skip = []

        for bdx in range(n_blocks):
            out_channels = block_channels[bdx]

            # first conv
            self.down_conv.append(conv_block(in_channels,out_channels,stride=1))
            self.is_down_skip.append(True)

            # second conv
            second_stride = 1 if bdx==(n_blocks-1) else 2
            self.down_conv.append(conv_block(out_channels,out_channels,stride=second_stride))
            self.is_down_skip.append(False)

            in_channels = out_channels
        #

        in_channels = block_channels[-1]
        self.up_conv = nn.ModuleList()
        self.is_up_skip = []

        for bdx in range(n_blocks-1):
            out_channels = block_channels[n_blocks-(bdx+2)]

            # first conv -> skip
            self.up_conv.append(conv_block(in_channels,in_channels,stride=1))
            self.is_up_skip.append(False)

            # second conv -> no skip
            self.up_conv.append(conv_block(2*in_channels,out_channels,stride=1))
            self.is_up_skip.append(True)

            # upsample
            self.up_conv.append(nn.Upsample(scale_factor=2))
            self.is_up_skip.append(False)

            in_channels = out_channels
        #

        self.field_layer = nn.Sequential(nn.Conv2d(block_channels[0],block_channels[0],kernel_size=3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(block_channels[0],d_out,kernel_size=1,padding=0))
    #

    def forward(self,input):
        out = input
        skip_reps = []
        # print(self.down_conv)
        for ldx,layer in enumerate(self.down_conv):
            out = layer(out)
            if self.is_down_skip[ldx]:
                skip_reps.append(out)
            #
        #

        sdx=1
        for ldx,layer in enumerate(self.up_conv):
            if self.is_up_skip[ldx]:
                skip_cat = torch.cat((out,skip_reps[-sdx]),dim=1)
                sdx+=1
                out = layer(skip_cat)
            #
            else:
                # print(type(layer))
                out = layer(out)
            #
        #

        return self.field_layer(out)
    #
#

'''
PatchGAN Discriminator
'''
class PatchDiscriminator(nn.Module):
    def __init__(self,d_in,d_out=0):
        super().__init__()

        def conv_block(n_in,n_out,use_bn):
            if use_bn:
                return nn.Sequential(nn.Conv2d(n_in,n_out,kernel_size=4,stride=2,padding=1),nn.BatchNorm2d(n_out),nn.LeakyReLU(negative_slope=.2,inplace=True))
            #
            else:
                return nn.Sequential(nn.Conv2d(n_in,n_out,kernel_size=4,stride=2,padding=1),nn.LeakyReLU(negative_slope=.2,inplace=True))
            #
        #

        # architecture from pix2pix
        # block_channels = [2,4,8,16,32,64,128,256,512]
        block_channels = [32,64,128,256,512]


        self.layers = nn.ModuleList()
        in_channels = d_in
        for bdx in range(len(block_channels)):
            out_channels = block_channels[bdx]
            self.layers.append(conv_block(in_channels,out_channels,bdx>0))
            in_channels = out_channels
        #

        # last layer: 1x1 conv to raw logits, then sigmoid
        # self.layers.append(nn.Sequential(nn.Conv2d(in_channels,1,kernel_size=1,padding=0),nn.Sigmoid()))
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels,1,kernel_size=1,padding=0)))
        self.layers.append(nn.Sequential(nn.AvgPool2d(int(block_channels[0]/2))))
    #

    def forward(self,input):
        out = input
        for ldx,layer in enumerate(self.layers):
            out = layer(out)
            print(out.shape)
        #

        return out
    #
#


'''
Image2value
'''

class Image2value2(nn.Module):
    def __init__(self,d_in,d_out=0):
        super().__init__()

        def conv_block(n_in,n_out,use_bn):
            if use_bn:
                return nn.Sequential(nn.Conv2d(n_in,n_out,kernel_size=4,stride=2,padding=1),nn.BatchNorm2d(n_out),nn.LeakyReLU(negative_slope=.2,inplace=True))
            #
            else:
                return nn.Sequential(nn.Conv2d(n_in,n_out,kernel_size=4,stride=2,padding=1),nn.LeakyReLU(negative_slope=.2,inplace=True))
            #
        #

        # architecture from pix2pix
        # block_channels = [2,4,8,16,32,64,128,256,512]
        # block_channels = [4,8,16,32,64,128,256,512]
        block_channels = [2,4,8,16,32,64,128,256,512]


        self.layers = nn.ModuleList()
        in_channels = d_in
        for bdx in range(len(block_channels)):
            out_channels = block_channels[bdx]
            self.layers.append(conv_block(in_channels,out_channels,bdx>0))
            in_channels = out_channels
        #

        # last layer: 1x1 conv to raw logits, then sigmoid
        # self.layers.append(nn.Sequential(nn.Conv2d(in_channels,1,kernel_size=1,padding=0),nn.Sigmoid()))
        # self.layers.append(nn.Sequential(nn.Conv2d(in_channels,1,kernel_size=1,padding=0)))
        # print(in_channels)
        # self.layers.append(nn.Sequential(nn.AvgPool2d(int(block_channels[0]/2))))
        self.layers.append(nn.Sequential(nn.Flatten()))
        #
        n_FC_in = int(in_channels*(512/(2**len(block_channels)))**2)
        # n_FC = int(in_channels/2)
        self.layers.append(nn.Sequential(nn.Linear(n_FC_in,n_FC_in),nn.LeakyReLU(negative_slope=.2,inplace=True)))
        self.layers.append(nn.Sequential(nn.Linear(n_FC_in,1)))
    #

    def forward(self,input):
        out = input
        for ldx,layer in enumerate(self.layers):
            out = layer(out)
            # print(out.shape)
        #

        return out
    #
#


if __name__=='__main__':
    # net = UNet(3)
    device = 'cuda'
    Pnet = Image2value2(1).to(device)
    # print(net)
    # print(Pnet)
    img = torch.rand(5,1,512,512).to(device)
    # Uout = net(img)
    Pout = Pnet(img)
    print(Pout.shape)

#
