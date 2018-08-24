import numpy as np
import torch
import torchvision

class Conv2D:
    # Class
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

    def forward(self, input_image):
        self.k1 = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.k2 = torch.tensor([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]])
        self.k3 = torch.tensor([[ 1,  1,  1], [1, 1, 1], [1, 1, 1]])
        self.k4 = torch.tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        self.k5 = torch.tensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])


        if self.mode == 'known':
            print "Known mode"
        else:
            print "Random mode"





