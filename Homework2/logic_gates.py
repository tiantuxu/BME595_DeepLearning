from neural_network import NeuralNetwork
import torch

class AND:
    def __init__(self):
        self.gate = NeuralNetwork([2, 1])
        self.theta = self.gate.getLayer(0)
        self.theta[0] = -3
        self.theta[1] = 2
        self.theta[2] = 2

    def __call__(self, x, y):
        self.x = x
        self.y = y
        output = self.forward()
        if output < 0.5:
            return False
        else:
            return True

    def forward(self):
        return self.gate.forward(torch.DoubleTensor([[self.x], [self.y]]))

class OR(NeuralNetwork):
    def __init__(self):
        self.gate = NeuralNetwork([2, 1])
        self.theta = self.gate.getLayer(0)
        self.theta[0] = -2
        self.theta[1] = 3
        self.theta[2] = 3

    def __call__(self, x, y):
        self.x = x
        self.y = y
        output = self.forward()
        if output < 0.5:
            return False
        else:
            return True

    def forward(self):
        return self.gate.forward(torch.DoubleTensor([[self.x], [self.y]]))

class NOT(NeuralNetwork):
    def __init__(self):
        self.gate = NeuralNetwork([1, 1])
        self.theta = self.gate.getLayer(0)
        self.theta[0] = 0
        self.theta[1] = -1

    def __call__(self, x):
        self.x = x
        output = self.forward()
        if output < 0.5:
            return False
        else:
            return True

    def forward(self):
        return self.gate.forward(torch.DoubleTensor([[self.x]]))

class XOR(NeuralNetwork):
    def __init__(self):
        self.gate = NeuralNetwork([2, 2, 1])
        self.theta1 = self.gate.getLayer(0)
        self.theta2 = self.gate.getLayer(1)

        self.theta1[0][0] = -2
        self.theta1[0][1] = -2
        self.theta1[1][0] = 3
        self.theta1[1][1] = -3
        self.theta1[2][0] = -3
        self.theta1[2][1] = 3

        self.theta2[0] = -2
        self.theta2[1] = 3
        self.theta2[2] = 3

    def __call__(self, x, y):
        self.x = x
        self.y = y
        output = self.forward()
        if output < 0.5:
            return False
        else:
            return True

    def forward(self):
        return self.gate.forward(torch.DoubleTensor([[self.x], [self.y]]))