#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    # YOUR CODE HERE for part 1f

    """
    Highway network implementation.
    """

    def __init__(self, input_dim):
        """
        Initializations.

        @param input_dim (int): Hidden size of x_conv_out.
        """

        super(Highway, self).__init__()
        self.proj_layer = nn.Linear(in_features=input_dim,
                                    out_features=input_dim,
                                    bias=True)
        self.gate_layer = nn.Linear(in_features=input_dim,
                                    out_features=input_dim,
                                    bias=True)

    def forward(self, x):
        """
        Forward computations.

        @param x (Tensor): tensor of size (batch_size, dim)

        @return x_highway (Tensor): tensor of the same shape as x
        """

        x_proj = self.proj_layer(x)
        x_proj = torch.relu(x_proj)
        x_gate = self.gate_layer(x)
        x_gate = torch.sigmoid(x_gate)

        x_highway = torch.mul(x_gate, x_proj) + torch.mul(1. - x_gate, x)
        return x_highway

    # END YOUR CODE


if __name__ == "__main__":
    highway = Highway(2)
    a = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.float)
    # I'm toooo lazy to do that
    print(highway(a).size())
