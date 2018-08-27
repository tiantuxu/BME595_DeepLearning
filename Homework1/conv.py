import numpy as np
import torch
import torchvision
from PIL import Image

class Conv2D:
    # Class
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

    def forward(self, input_image):
        self.input_image = input_image
        self.k1 = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.k2 = torch.tensor([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]])
        self.k3 = torch.tensor([[ 1,  1,  1], [1, 1, 1], [1, 1, 1]])
        self.k4 = torch.tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        self.k5 = torch.tensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])

        image_height = input_image.shape[1]
        image_width = input_image.shape[2]
        print image_width, image_height

        image_row = int((image_height - self.kernel_size)/self.stride + 1)
        image_col = int((image_width - self.kernel_size)/self.stride + 1)
        output_tensor = torch.zeros((image_height - self.kernel_size)/self.stride + 1, (image_width - self.kernel_size)/self.stride + 1, self.o_channel)


        kernel = []

        if self.mode == 'known':
            if self.o_channel == 1:
                print("Task1")
                kernel.append(torch.stack([self.k1 for i in range(self.in_channel)]))

                for k_count in range(0, self.o_channel):
                    Number_of_ops = 0

                    for i in range(0, image_row):
                        for j in range(0, image_col):
                            out = torch.mul(kernel[k_count].float(),
                                            self.input_image[:, i * self.stride : i * self.stride + self.kernel_size,
                                            j * self.stride : j * self.stride + self.kernel_size])
                            Number_of_ops += self.kernel_size * self.kernel_size * self.in_channel
                            output_tensor[i][j] = out.sum()
                            Number_of_ops += self.kernel_size * self.kernel_size * self.in_channel - 1

                    print ("Task 1: Total operations for k" + str(k_count) + " is " + str(Number_of_ops))

                    return Number_of_ops, output_tensor

            elif self.o_channel == 2:
                print("Task2")
                kernel.append(torch.stack([self.k4 for i in range(self.in_channel)]))
                kernel.append(torch.stack([self.k5 for i in range(self.in_channel)]))

                for k_count in range(0, self.o_channel):
                    Number_of_ops = 0

                    for i in range(0, image_row):
                        for j in range(0, image_col):
                            out = torch.mul(kernel[k_count].float(),
                                            self.input_image[:, i * self.stride: i * self.stride + self.kernel_size,
                                            j * self.stride: j * self.stride + self.kernel_size])
                            Number_of_ops += self.kernel_size * self.kernel_size * self.in_channel
                            output_tensor[i][j][k_count] = out.sum()
                            Number_of_ops += self.kernel_size * self.kernel_size * self.in_channel - 1
                    print ("Task 2: Total operations for k" + str(k_count) + " is " + str(Number_of_ops))


                return Number_of_ops, output_tensor
            else:
                print("Task3")
                kernel.append(torch.stack([self.k1 for i in range(self.in_channel)]))
                kernel.append(torch.stack([self.k2 for i in range(self.in_channel)]))
                kernel.append(torch.stack([self.k3 for i in range(self.in_channel)]))
                for k_count in range(0, self.o_channel):
                    Number_of_ops = 0

                    for i in range(0, image_row):
                        for j in range(0, image_col):
                            out = torch.mul(kernel[k_count].float(),
                                            self.input_image[:, i * self.stride: i * self.stride + self.kernel_size,
                                            j * self.stride: j * self.stride + self.kernel_size])
                            Number_of_ops += self.kernel_size * self.kernel_size * self.in_channel
                            output_tensor[i][j][k_count] = out.sum()
                            Number_of_ops += self.kernel_size * self.kernel_size * self.in_channel - 1
                    print ("Task 3: Total operations for k" + str(k_count) + " is " + str(Number_of_ops))

                return Number_of_ops, output_tensor
        else:
            #print "Random mode"
            for out_count in range(0, self.o_channel):
                rand_kernel = torch.randn( self.in_channel, self.kernel_size, self.kernel_size)
                Number_of_ops = 0

                for i in range(0, image_row):
                    for j in range(0, image_col):
                        out = torch.mul(rand_kernel.float(),
                                        self.input_image[:, i * self.stride: i * self.stride + self.kernel_size,
                                        j * self.stride: j * self.stride + self.kernel_size])
                        Number_of_ops += self.kernel_size * self.kernel_size * self.in_channel
                        output_tensor[i][j][out_count] = out.sum()
                        Number_of_ops += self.kernel_size * self.kernel_size * self.in_channel - 1

            return Number_of_ops, output_tensor








