#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    # YOUR CODE HERE for part 1g
    """
    CNN implementation.
    """

    def __init__(self, in_chan, out_chan, kernel_size=5, pad_size=1):
        """
        Initializations.

        @param in_chan (int): input channels.
        @param out_chan (int): output channels.
        @param kernel_size (int): window size.
        """
        super(CNN, self).__init__()
        self.conv_1d = nn.Conv1d(in_channels=in_chan,
                                 out_channels=out_chan,
                                 kernel_size=kernel_size,
                                 padding=pad_size)

    def forward(self, x):
        """
        Forward, from x_reshape to x_conv_out.

        @param x (tensor): x_reshape

        @return x_conv_out (tensor): x_conv_out
        """
        x_conv = self.conv_1d(x)
        x_conv = torch.relu(x_conv)
        x_conv_out = nn.functional.adaptive_max_pool1d(x_conv, 1).squeeze(2)
        return x_conv_out

    # END YOUR CODE
