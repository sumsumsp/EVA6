import torch.nn as nn
import torch.nn.functional as F

def buildConvLayer(in_channels, out_channels, kernel_size = 3, padding = 0, bias = False, activation = nn.ReLU ,normalization = None, group_count = 2, dropout = None):
    conv_layer = []

    conv_layer.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding, bias=bias))
    conv_layer.append(activation())

    if normalization:
        if "BN" == normalization:
            conv_layer.append(nn.BatchNorm2d(out_channels))
        if "GN" == normalization:
            conv_layer.append(nn.GroupNorm(group_count,out_channels))
        if "LN" == normalization:
            conv_layer.append(nn.GroupNorm(1,out_channels))

    if dropout:
        conv_layer.append(nn.Dropout(dropout))
    
    return conv_layer
      

def buildConvBlock(in_channels, out_channels_list, kernel_size = 3, padding = 0, bias = False, activation = nn.ReLU ,normalization = None, group_count = 2, dropout = None, dropout_layers = 'last'):
    conv_block = []
    dropout_val = None

    if dropout and 'all' == dropout_layers:
        dropout_val = dropout

    for out_channels in out_channels_list:
        conv_block += buildConvLayer(in_channels, out_channels, kernel_size, padding, bias, activation, normalization, group_count, dropout_val)
        in_channels = out_channels

    if dropout and 'last' == dropout_layers:
        conv_block.append(nn.Dropout(dropout))

    return nn.Sequential(*conv_block)


def buildTransBlock(in_channels, out_channels):
    trans_block = []

    if in_channels != out_channels:
        trans_block.append(buildConvLayer(in_channels, out_channels, kernel_size=1))

    trans_block.append(nn.AvgPool2d(2, 2))

    return nn.Sequential(*trans_block)
     


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()

        self.conv1 = buildConvBlock(1,[10,10,10],**kwargs)
        self.trans1 = buildTransBlock(10,10)
        self.conv2 = buildConvBlock(10,[10,10],**kwargs)
        self.trans2 = buildTransBlock(10,10)
        self.layers = [self.conv1, self.trans1, self.conv2, self.trans2]       
        self.fc = nn.Linear(10*3*3,10)

    def forward(self, x):
        for f in self.layers:
          x = f(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return F.log_softmax(x,dim=1)
