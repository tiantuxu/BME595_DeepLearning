import numpy as np
import torch
import torchvision
from scipy import misc

input_image = []
input_image.append(misc.imread(image0.png))
input_image.append(misc.imread(image1.png))

# Part A, Task 1
conv2d = Conv2D(in_channel=int(3), o_channel=int(1), kernel_size=int(3), stride=int(1),mode='known')

for i in range(2):
    [Number_of_ops, output_image] = conv2d.forward(input_image[j])
    print ("Total operations for image%d is %d", i, Number_of_ops)
    img_name = "image" + str(i) + "_task1_k1.png"
    misc.imsave(img_name, output_image[:,:,0])

# Part A, Task 2
conv2d = Conv2D(in_channel=int(3), o_channel=int(2), kernel_size=int(5), stride=int(1),mode='known')

for i in range(2):
    [Number_of_ops, output_image] = conv2d.forward(input_image[j])
    for j in range(output_image.length()):
        print ("Total operations for image%d is %d", i, Number_of_ops)
        img_name = "image" + str(i) + "_task1_k" + str(j) + ".png"
        misc.imsave(img_name, output_image[:,:,0])

# Part A, Task 3
conv2d = Conv2D(in_channel=int(3), o_channel=int(3), kernel_size=int(3), stride=int(2),mode='known')
