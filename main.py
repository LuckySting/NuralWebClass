import math
import random

import numpy as np

SPEED = 0.5
BIAS = 1
AGES = 5000
SCHEME = [2, 4, 1]
X = [[0, 0], [1, 0], [0, 1], [1, 1]]
Y = [0, 1, 1, 0]


def sigmoid(x, d=False):
    if (d):
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + math.exp(-x))


class Neuron:
    def __init__(self, weights_count):
        self.inputs = None
        self.output = None
        self.output_d = None
        self.delta = None
        self.weights = [(random.randint(0, 100) - 50) / 100 for i in range(weights_count)]

    def feed(self, inputs):
        self.inputs = inputs
        weighted_sum = BIAS + sum([self.inputs[i] * self.weights[i] for i in range(len(inputs))])
        self.output = sigmoid(weighted_sum)
        self.output_d = sigmoid(weighted_sum, d=True)

    def backward(self, delta):
        self.delta = delta
        return [w * delta for w in self.weights]

    def train(self):
        for i in range(len(self.weights)):
            self.weights[i] += SPEED * self.delta * self.output_d * self.inputs[i]


class Layer:
    def __init__(self, neurons_count, weights_count):
        self.neurons = [Neuron(weights_count) for _ in range(neurons_count)]

    def feed(self, inputs):
        for neuron in self.neurons:
            neuron.feed(inputs)

    def backward(self, weighted_deltas):
        next_w_d = []
        for n in range(len(self.neurons)):
            delta = sum([d[n] for d in weighted_deltas])
            next_w_d.append(self.neurons[n].backward(delta))
        return next_w_d

    def train(self):
        for n in self.neurons:
            n.train()


class Network:
    def __init__(self, scheme):
        self.layers = [Layer(scheme[0], 0)]
        for layer in range(len(scheme) - 1):
            self.layers.append(Layer(scheme[layer + 1], scheme[layer]))

    def feed(self, inputs):
        # заряжаем первый слой
        for i in range(len(self.layers[0].neurons)):
            self.layers[0].neurons[i].output = inputs[i]

        # вход следующего это выход текущего
        for i in range(len(self.layers) - 1):
            outputs = [neuron.output for neuron in self.layers[i].neurons]
            self.layers[i + 1].feed(outputs)

        return self.layers[-1].neurons[0].output

    def backward(self, error):
        weighted_deltas = self.layers[-1].backward(error)
        for l in range(len(self.layers) - 2, 0, -1):
            weighted_deltas = self.layers[l].backward(weighted_deltas)

    def train(self, xs, ys, ages):
        for _ in range(ages):
            errors = []
            for i in range(len(xs)):
                z = self.feed(xs[i])
                error = ys[i] - z
                errors.append(abs(error))
                self.backward([[error]])
                for l in self.layers:
                    l.train()
            print("ошибка: {}".format(np.mean(errors)))


test = Network(SCHEME)
test.train(X, Y, AGES)

print([test.feed([0, 0]), test.feed([0, 1]), test.feed([1, 0]), test.feed([1, 1])])
