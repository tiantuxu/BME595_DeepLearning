import os, sys
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from time import time


class LeNet5(nn.Module):
    def __init__(self):

        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.l1 = nn.Linear(16 * 5 * 5, 120)
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, 10)

    def forward(self, x):
        out = torch.nn.functional.relu(self.conv1(x))
        out = torch.nn.functional.max_pool2d(out, 2)
        out = torch.nn.functional.relu(self.conv2(out))
        out = torch.nn.functional.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.relu(self.l1(out))
        out = torch.nn.functional.relu(self.l2(out))
        out = self.l3(out)
        return out

class NNImg2Num:

    def __init__(self):

        self.outcomes = 10
        self.learning_rate = 1.0
        self.epochs = 10
        self.batch = 10

        self.input_size = 28 * 28  # input image size

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.batch, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.batch, shuffle=True)

        self.model = LeNet5()

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
                #output = self.model(data.view(self.batch, self.input_size))
                output = self.model(data)
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
                output = self.model(data)
                loss = self.loss(output, onehot_target)
                value, index = torch.max(output, 1)

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
            print("Epoch = " + str(i + 1) + ", Validation loss = " + str(float(validation_loss)/(len(self.test_loader.dataset) / self.batch)) + " , Validation Accuracy = " + str(float(accuracy)))

    def forward(self, img):
        self.model.eval()
        output = self.model(img.view(1, self.input_size))
        value, index = torch.max(output, 1)
        return int(index)


if __name__ == '__main__':
    nnimg2num = NNImg2Num()
    nnimg2num.train()

