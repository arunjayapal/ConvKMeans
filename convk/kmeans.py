"""A PyTorch implementation of the Convolutional K-means.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class ConvKMeans(nn.Module):
    """ConvKMeans."""

    def __init__(self, input_shape, out_channels, kernel_size, lr=1.,
                 stride=1, padding="valid", groups=1, bias=False):
        """Conv K-Means."""
        super(ConvKMeans, self).__init__()

        # set all parameters
        self.input_shape = input_shape
        self.in_channels = input_shape[1]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.lr = lr
        if padding == "valid":
            self.padding = 0
        elif padding == "same":
            self.padding = (kernel_size[0] - 1)//2
            self.zero_pad = nn.ZeroPad2d(self.padding)
        self.groups = groups
        self.bias = bias
        self.batch_size = input_shape[0]
        self.back_kernel_size = (input_shape[2]-kernel_size[0]+1,
                                 input_shape[3]-kernel_size[1]+1)

        # create parameter
        self.kernel = Variable(
            torch.Tensor(self.out_channels,
                         self.in_channels // self.groups,
                         *self.kernel_size))
        if self.bias:
            self.bias = Variable(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # forward pass
        feature_out = F.conv2d(x, self.kernel,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=1,
                               groups=self.groups)

        # select the highest
        hid_out = feature_out*(feature_out >= feature_out.max(
            1, keepdim=True)[0]).float()
        # swap hidden output to be back kernel
        hid_out = hid_out.permute(1, 0, 2, 3)
        new_x = x.permute(1, 0, 2, 3)
        bound = (hid_out.size()[2]-1)*self.stride + \
            self.kernel_size[0]
        if self.padding != 0:
            new_x = self.zero_pad(new_x)
        new_x = new_x[:, :, :bound, :bound]

        kernel_out = F.conv2d(new_x, hid_out,
                              stride=1,
                              padding=0,
                              dilation=self.stride,
                              groups=self.groups)
        kernel_out = kernel_out.permute(1, 0, 2, 3)

        # update kernel
        self.kernel += self.lr*kernel_out

        # normalize kernel
        norm = self.kernel.view(self.kernel.size()[0], -1).norm(dim=1)
        norm = norm.view(norm.size()[0], 1, 1, 1).expand_as(self.kernel)
        self.kernel = self.kernel/norm

        return feature_out
