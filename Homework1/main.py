import numpy as np
import io
import torch
import torchvision
from conv import Conv2D
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from scipy import misc
import time
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

toTensor = transforms.Compose([transforms.ToTensor()])
input_image = []
input_image.append(Image.open("image0.jpg"))
input_image.append(Image.open("image1.jpg"))

# Part A, Task 1
conv2d = Conv2D(in_channel=int(3), o_channel=int(1), kernel_size=int(3), stride=int(1),mode='known')
for i in range(2):
    [Number_of_ops, output_image] = conv2d.forward(toTensor(input_image[i]))
    img_name = "image" + str(i) + "_task1_k1.jpg"
    misc.imsave(img_name, output_image[:,:,0])


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

# Part B
print("Part B")
index = [i for i in range(11)]
fig = plt.figure(figsize=(20, 10))

for img_count in range(2):
    total_time = []

    for i in range(11):
        conv2d = Conv2D(in_channel=3, o_channel= 2**i, kernel_size= 3, stride=1, mode='rand')
        stime = time.time()
        [Number_of_ops, output_image] = conv2d.forward(toTensor(input_image[img_count]))
        total_time.append(time.time() - stime)

    if(img_count == 0):
        plt.plot(index, total_time, c="red", linewidth=2.0, label='1280x720')
    else:
        plt.plot(index, total_time, c="blue", linewidth=2.0, label='1920x1080')

plt.xlabel("i", fontsize=20)
plt.ylabel("Total time for convolution", fontsize=20)
plt.title("Total time on input images")
plt.legend()
plt.savefig("Part-B")
plt.close()

# Part C
print("Part C")
index = [2*i+3 for i in range(5)]
fig = plt.figure(figsize=(20, 10))

for img_count in range(2):
    operations = []

    for i in range(5):
        conv2d = Conv2D(in_channel=3, o_channel= 2, kernel_size= 2 * i + 3, stride=1, mode='rand')
        [Number_of_ops, output_image] = conv2d.forward(toTensor(input_image[img_count]))
        operations.append(Number_of_ops)

    if(img_count == 0):
        plt.plot(index, operations, c="red", linewidth=2.0, label='1280x720')
    else:
        plt.plot(index, operations, c="blue", linewidth=2.0, label='1920x1080')

plt.xlabel("i", fontsize=20)
plt.ylabel("Total operations", fontsize=20)
plt.title("Total operations on input images")
plt.legend()
plt.savefig("Part-C")
plt.close()
