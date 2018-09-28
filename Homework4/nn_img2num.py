#!/usr/bin/env python3

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from time import time


class NnImg2Num:
    def __init__(self):

        self.outcomes = 10  # number of possible outcomes
        self.learning_rate = 15  # learning rate
        self.epochs = 10  # default epoch
        self.batch = 10

        self.input_size = 28 * 28  # input image size

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.batch, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.batch, shuffle=True)

        self.model = nn.Sequential(nn.Linear(self.input_size, 392), nn.Sigmoid(),
                                   nn.Linear(392, 196), nn.Sigmoid(),
                                   nn.Linear(196, 98), nn.Sigmoid(),
                                   nn.Linear(98, 49), nn.Sigmoid(),
                                   nn.Linear(49, self.outcomes), nn.Sigmoid())

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.loss = nn.MSELoss()

    def train(self):

        def train_epoch():
            self.model.train()
            training_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                onehot_target = torch.zeros(self.batch, self.outcomes)
                for n in range(len(target)):
                    onehot_target[n][target[n]] = 1

                self.optimizer.zero_grad()
                output = self.model(data.view(self.batch, self.input_size))
                loss = self.loss(output, onehot_target)
                training_loss += loss.data.item()
                loss.backward()
                self.optimizer.step()
            return training_loss
            #print("Train epoch " + str(i+1) + ", loss = " + str(float(training_loss)/(len(self.train_loader.dataset)/self.batch)))

        def validation():
            self.model.eval()
            true_positive = 0
            validation_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                onehot_target = torch.zeros(self.batch, self.outcomes)
                for n in range(len(target)):
                    onehot_target[n][target[n]] = 1
                output = self.model(data.view(self.batch, self.input_size))
                value, index = torch.max(output, 1)
                loss = self.loss(output, onehot_target)

                validation_loss += float(loss.data)

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

            print("Epoch = " + str(i + 1) + ", Training loss = " + str(float(training_loss) / (len(self.train_loader.dataset) / self.batch)) + ", time =  " + str(float(total_time)))
            print("Epoch = " + str(i + 1) + ", Validation loss = " + str(float(validation_loss) / (len(self.test_loader.dataset) / self.batch)) + " , Validation Accuracy = " + str(float(accuracy)))

    def forward(self, img):
        self.model.eval()
        output = self.model(img.view(1, self.input_size))
        value, index = torch.max(output, 1)
        return int(index)


if __name__ == '__main__':
    nnimg2num = NnImg2Num()
    nnimg2num.train()

