# import the necessary packages
from sympy.solvers.diophantine.diophantine import length
from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
import torch

        
class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        # This is the basic block of the network
        return self.conv2(self.relu(self.conv1(x)))

class SimpleConv(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64),
                 dec_channels=(64, 32, 16, 8)):
        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        # prepare pooling
        self.pooling = MaxPool2d(kernel_size=2, stride=2)

        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.dec_blocks = ModuleList(
            [Block(dec_channels[i + 1], dec_channels[i + 1])
             for i in range(len(dec_channels) - 1)])

        # prepare upconvolutions
        self.upconv = ModuleList([
            ConvTranspose2d(dec_channels[i], dec_channels[i + 1], kernel_size=2, stride=2)
            for i in range(len(dec_channels) - 1)])
                # Last conv to get 1 channel
        self.last_conv = Conv2d(in_channels=self.channels[-1], out_channels=1, kernel_size=1)


    def forward(self, x):
        # loop through the encoder blocks
        for block in self.enc_blocks:
            x = block(x)
            # pooling: halve the spatial resolution
            x = self.pooling(x)

        # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            # upconvolution: change channels and double the spatial resolution
            x = self.upconv[i](x)
            # decoder block (no concatenation, so no extra channels)
            x = self.dec_blocks[i](x)

        # return the final decoder output
        x = self.last_conv(x)     
        return x

class UNet(Module):

    # def __init__(self, enc_channels=(3, 16, 32, 64),
    #              dec_channels=(64, 32, 16, 8)):
    def __init__(self, num_features = 16, depth_enc = 3):
        # Dec_channels are different from simple convolution, it ends at 8 before applying the last conv
        super().__init__()
        # Compute input and output channels based on number of features and depth
        enc_channels = [num_features*(2**n) for n in range(depth_enc)]
        dec_channels = enc_channels[::-1]
        dec_channels.append(dec_channels[-1]//2)
        enc_channels.insert(0,3)
        # Enc_channels define the progression of feature channels
        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        # -- prepare pooling
        self.pooling = MaxPool2d(kernel_size=2, stride=2)
        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels

        # In this way, it already takes into account the concatenation of features (sum of channels)
        self.dec_blocks = ModuleList([
            Block(dec_channels[i + 1] + enc_channels[len(enc_channels) - 1 - i], dec_channels[i + 1])
            for i in range(len(dec_channels) - 1)
        ])

        # -- prepare upconvolutions 
        self.upconv = ModuleList([
            ConvTranspose2d(dec_channels[i], dec_channels[i+1], kernel_size=2, stride=2)
            for i in range(len(dec_channels) - 1)])
        self.bottleneck = Block(enc_channels[-1], enc_channels[-1]) 
        # Last conv to get 1 channel
        self.last_conv = Conv2d(in_channels=self.channels[-1], out_channels=1, kernel_size=1)

    def forward(self, x):
        # loop through the encoder blocks
        # we have to store the outputs of the encoder blocks
        skip_connections = []  # these will be concatenated to the decoding part
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            # first, the results are stored
            skip_connections.append(x)
            # then apply pooling
            x = self.pooling(x)
        # process latent space
        x = self.bottleneck(x)
        # Now x is the latent space
        # decoder: loop through the number of channels
        # now, from bottom to up: we concatenate the upsampled with the encoder results
        skip_connections.reverse()
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            # TODO: upconvolutions
            x = self.upconv[i](x) # Upconvolution
            # Apply the skip connections:  concat
            x = torch.cat((x, skip_connections[i]), dim=1)
            x = self.dec_blocks[i](x)  # forward pass of decoder block for x

        # Last conv to get 1 channel
        x = self.last_conv(x)

        return x
        
class HourGlass(Module):
    # def __init__(self, enc_channels=(3, 16, 32, 64),
    #              dec_channels=(64, 32, 16, 8)):
    def __init__(self, num_features = 16, depth_enc = 3):
        # Dec_channels are different from simple convolution, it ends at 8 before applying the last conv
        super().__init__()
        # Compute input and output channels based on number of features and depth
        enc_channels = [num_features*(2**n) for n in range(depth_enc)]
        dec_channels = enc_channels[::-1]
        dec_channels.append(dec_channels[-1]//2)
        enc_channels.insert(0,3)

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        # TODO: prepare pooling
        self.pool = MaxPool2d(2, stride=2, padding=0)
        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.dec_blocks = ModuleList(
            [Block(dec_channels[i + 1], dec_channels[i + 1])
             for i in range(len(dec_channels) - 1)])
        # TODO: prepare upconvolutions
        self.upconv = ModuleList(
            [ConvTranspose2d(dec_channels[i], dec_channels[i + 1], 2, stride=2)
             for i in range(len(dec_channels) - 1)])
         # Last conv to get 1 channel
        self.last_conv = Conv2d(in_channels=self.channels[-1], out_channels=1, kernel_size=1)

    
    def forward(self, x):
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            # TODO: pooling
            x = self.pool(x)

        # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            # TODO: upconvolutions
            x = self.upconv[i](x)
            x = self.dec_blocks[i](x)
        x = self.last_conv(x)

        # return the final decoder output
        return x


    
