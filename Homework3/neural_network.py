import torch
import numpy as np
import math

class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.input_layer = layers[0]
        self.output_layer = layers[-1]
        self.layers = layers

        self.Theta = {}
        self.key = []

        self.dE_dTheta = {}
        self.a = {}
        self.z = {}

        for i in range(self.num_layers - 1):
            self.key.append(str(i) + "-" + str(i + 1))

        for i in range(self.num_layers - 1):
            if i == 0:
                self.Theta[self.key[i]] = torch.normal(mean=torch.zeros(self.layers[i+1], self.input_layer + 1), std=torch.zeros(self.layers[i+1], self.input_layer + 1).fill_(1.0/math.sqrt(self.input_layer + 1))).type(torch.FloatTensor)
            else:
                self.Theta[self.key[i]] = torch.normal(mean=torch.zeros(self.layers[i+1], self.layers[i] + 1), std=torch.zeros(self.layers[i+1], self.layers[i] + 1).fill_(1.0/math.sqrt(self.layers[i] + 1))).type(torch.FloatTensor)

        self.loss = 1

    def getLayer(self, layer):
        return self.Theta[self.key[layer]]

    def forward(self, input):
        self.input = input
        [x, y] = input.shape
        self.a[0] = self.input
        bias = torch.ones((1, y)).type(torch.FloatTensor)
        for i in range(0, self.num_layers - 1):
            self.a[i] = torch.cat((bias, self.a[i]), 0)
            self.z[i + 1] = torch.mm(self.Theta[self.key[i]], self.a[i])
            self.a[i + 1] = 1.0 / (1.0 + np.exp(-self.z[i + 1]))

        return self.a[self.num_layers - 1]

    def backward(self, target, loss):
        self.target = target
        self.loss = loss

        (r, c) = self.target.size()

        self.total_loss = ((self.a[self.num_layers - 1] - self.target).pow(2).sum()) / (2 * c)

        if loss == 'MSE':
            '''delta = (a - y)d(sigmoid) = (a-y)*(a(1-a))'''
            delta = torch.mul(self.a[self.num_layers - 1] - self.target, self.a[self.num_layers - 1] * (1 - self.a[self.num_layers - 1]))
            for i in range(self.num_layers - 2, -1, -1):
                if i == self.num_layers - 2:
                    self.dE_dTheta[i] = torch.mm(delta, self.a[i].t())
                else:
                    delta = delta[1:len(delta), :]
                    self.dE_dTheta[i] = torch.mm(delta, self.a[i].t())

                tmp = torch.mm(self.Theta[self.key[i]].t(), delta)
                delta = torch.mul(tmp, self.a[i] * (1 - self.a[i]))

    def updateParams(self, eta):
        self.eta = eta
        for i in range(self.num_layers - 1):
            self.Theta[self.key[i]] -= torch.mul(self.dE_dTheta[i], self.eta)
