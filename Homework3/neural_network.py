import torch
import numpy as np
import math

class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.input_layer = layers[0]
        self.output_layer = layers[-1]
        self.layers = layers

        self.theta = {}
        self.key = []
        for i in range(self.num_layers - 1):
            self.key.append(str(i) + "-" + str(i + 1))

        for i in range(self.num_layers - 1):
            if i == 0:
                self.theta[self.key[i]] = torch.normal(mean=torch.zeros(self.input_layer + 1, self.layers[i+1]), std=torch.zeros(self.input_layer + 1, self.layers[i+1]).fill_(1/math.sqrt(self.layers[i+1]))).type(torch.DoubleTensor)
            else:
                self.theta[self.key[i]] = torch.normal(mean=torch.zeros(self.layers[i] + 1, self.layers[i+1]), std=torch.zeros(self.layers[i] + 1, self.layers[i+1]).fill_(1/math.sqrt(self.layers[i+1]))).type(torch.DoubleTensor)

    def getLayer(self, layer):
        return self.theta[self.key[layer]]

    def forward(self, input):
        self.input = input
        [x, y] = input.shape
        output = self.input
        bias = torch.ones((1, y)).type(torch.DoubleTensor)
        for i in range(self.num_layers - 1):
            in_mtx = torch.cat((bias, output), 0)
            theta_t = torch.t(self.theta[self.key[i]])
            out_mtx = torch.mm(theta_t, in_mtx)
            output = 1.0 / (1.0 + np.exp(-out_mtx))

        return output
