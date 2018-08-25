import numpy as np
import io
import torch
import torchvision
from conv import Conv2D
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from scipy import misc
toTensor = transforms.Compose([transforms.ToTensor()])
input_image = []
input_image.append(Image.open("image0.jpg"))
input_image.append(Image.open("image1.jpg"))
'''
# Part A, Task 1
conv2d = Conv2D(in_channel=int(3), o_channel=int(1), kernel_size=int(3), stride=int(1),mode='known')

for i in range(2):
    [Number_of_ops, output_image] = conv2d.forward(toTensor(input_image[i]))
    img_name = "image" + str(i) + "_task1_k1.jpg"
    misc.imsave(img_name, output_image[:,:,0])
'''

# Part A, Task 2
conv2d = Conv2D(in_channel=int(3), o_channel=int(2), kernel_size=int(5), stride=int(1),mode='known')
for i in range(2):
    [Number_of_ops, output_image] = conv2d.forward(toTensor(input_image[i]))
    for j in range(2):
        img_name = "image" + str(i) + "_task2_k" + str(j) + ".jpg"
        misc.imsave(img_name, output_image[:,:,j])

# Part A, Task 3
conv2d = Conv2D(in_channel=int(3), o_channel=int(3), kernel_size=int(3), stride=int(2),mode='known')
for i in range(2):
    [Number_of_ops, output_image] = conv2d.forward(toTensor(input_image[i]))
    for j in range(3):
        img_name = "image" + str(i) + "_task3_k" + str(j) + ".jpg"
        misc.imsave(img_name, output_image[:,:,j])
