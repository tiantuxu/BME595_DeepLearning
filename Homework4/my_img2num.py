#!/usr/bin/env python3

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from time import time
from neural_network import NeuralNetwork


class MyImg2Num:
    def __init__(self):
        self.outcomes = 10
        self.learning_rate = 0.1
        self.epochs = 10
        self.batch = 10

        self.input_size = 28 * 28

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.batch, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.batch, shuffle=True)

        self.model = NeuralNetwork([self.input_size, 392, 196, 98, 49, self.outcomes])

    def train(self):
        def train_epoch():
            training_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                onehot_target = torch.zeros(self.batch, self.outcomes)
                for n in range(len(target)):
                    onehot_target[n][target[n]] = 1

                self.model.forward(data.view(self.batch, self.input_size))
                self.model.backward(onehot_target, 'MSE')

                training_loss += self.model.total_loss

                self.model.updateParams(self.learning_rate)

            #print "loss = ", float(training_loss)/(len(self.train_loader.dataset)/self.batch)
            return training_loss

        def validation():
            true_positive = 0
            validation_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                onehot_target = torch.zeros(self.batch, self.outcomes)

                for n in range(len(target)):
                    onehot_target[n][target[n]] = 1

                output = self.model.forward(data.view(self.batch, self.input_size))
                value, index = torch.max(output, 1)

                validation_loss += (onehot_target - output).pow(2).sum()/2.0

                for n in range(len(target)):
                    if index[i] == target[i]:
                        true_positive += 1

            return float(true_positive)/len(self.test_loader.dataset), validation_loss

        for i in range(self.epochs):
            start = time()
            training_loss = train_epoch()
            end = time()
            total_time = end - start
            accuracy, validation_loss = validation()
            print("Epoch = " + str(i+1) + ", Training loss = " + str(float(training_loss)/(len(self.train_loader.dataset)/self.batch)) + ", time =  " + str(float(total_time)))
            print("Epoch = " + str(i+1) + ", Validation loss = " + str(float(validation_loss)/len(self.test_loader.dataset)) + " , Validation Accuracy = " + str(float(accuracy)))

    def forward(self, img):
        self.model.eval()
        output = self.model.forward(img.view(1, self.input_size))
        value, index = torch.max(output, 1)
        return int(index)


if __name__ == '__main__':
    myimg2num = MyImg2Num()
    myimg2num.train()
