from neural_network import NeuralNetwork
import torch
from random import randint


class AND:
    def __init__(self):
        self.gate = NeuralNetwork([2, 1])
        #self.theta = self.gate.getLayer(0)
        #self.theta[0] = -3
        #self.theta[1] = 2
        #self.theta[2] = 2
        self.iter = 1000

    def __call__(self, x, y):
        self.x = x
        self.y = y
        output = self.forward()
        if output < 0.5:
            return False
        else:
            return True

    def forward(self):
        return self.gate.forward(torch.FloatTensor([[self.x], [self.y]]))

    def train(self):
        for i in range(self.iter):
            x = randint(0, 1)
            y = randint(0, 1)
            target = torch.FloatTensor([[x and y]])
            output = self.gate.forward(torch.FloatTensor([[x], [y]]))
            self.gate.backward(target, 'MSE')
            self.gate.updateParams(1)

        print self.gate.getLayer(0)

class OR(NeuralNetwork):
    def __init__(self):
        self.gate = NeuralNetwork([2, 1])
        #self.theta = self.gate.getLayer(0)
        #self.theta[0] = -2
        #self.theta[1] = 3
        #self.theta[2] = 3
        self.iter = 1000


    def __call__(self, x, y):
        self.x = x
        self.y = y
        output = self.forward()
        if output < 0.5:
            return False
        else:
            return True

    def forward(self):
        return self.gate.forward(torch.FloatTensor([[self.x], [self.y]]))

    def train(self):
        for i in range(self.iter):
            x = randint(0, 1)
            y = randint(0, 1)
            target = torch.FloatTensor([[x or y]])
            output = self.gate.forward(torch.FloatTensor([[x], [y]]))
            self.gate.backward(target, 'MSE')
            self.gate.updateParams(1)

        print self.gate.getLayer(0)


class NOT(NeuralNetwork):
    def __init__(self):
        self.gate = NeuralNetwork([1, 1])
        #self.theta = self.gate.getLayer(0)
        #self.theta[0] = 0
        #self.theta[1] = -1
        self.iter = 1000

    def __call__(self, x):
        self.x = x
        output = self.forward()
        if output < 0.5:
            return False
        else:
            return True

    def forward(self):
        return self.gate.forward(torch.FloatTensor([[self.x]]))

    def train(self):
        for i in range(self.iter):
            x = randint(0, 1)
            target = torch.FloatTensor([[not x]])
            output = self.gate.forward(torch.FloatTensor([[x]]))
            self.gate.backward(target, 'MSE')
            self.gate.updateParams(1)

        print self.gate.getLayer(0)


class XOR(NeuralNetwork):
    def __init__(self):
        self.gate = NeuralNetwork([2, 2, 1])
        #self.theta1 = self.gate.getLayer(0)
        #self.theta2 = self.gate.getLayer(1)
        self.iter = 100000

        '''
        self.theta1[0][0] = -2
        self.theta1[0][1] = -2
        self.theta1[1][0] = 3
        self.theta1[1][1] = -3
        self.theta1[2][0] = -3
        self.theta1[2][1] = 3

        self.theta2[0] = -2
        self.theta2[1] = 3
        self.theta2[2] = 3
        '''

    def __call__(self, x, y):
        self.x = x
        self.y = y
        output = self.forward()
        if output < 0.5:
            return False
        else:
            return True

    def forward(self):
        return self.gate.forward(torch.FloatTensor([[self.x], [self.y]]))

    def train(self):
        for i in range(self.iter):
            x = randint(0, 1)
            y = randint(0, 1)
            target = torch.FloatTensor([[float((x and (not y)) or ((not x) and y))]])
            output = self.gate.forward(torch.FloatTensor([[x], [y]]))
            self.gate.backward(target, 'MSE')
            self.gate.updateParams(0.05)

        print self.gate.getLayer(0)
        print self.gate.getLayer(1)
