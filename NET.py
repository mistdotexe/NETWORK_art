from array import array
from audioop import bias
from math import exp
import numpy as np
import gnumpy as gnp

weights = []
biases = []
weight_range = 4
bias_range = 1
input_dim = 1
size = 1, 1


def activation_func(x):
    return np.tanh(x)


activation_func_vec = np.vectorize(activation_func)


def gen_weights() -> array:
    arr = []
    global weights
    for n in range(len(size)-1):
        tmp_weights = np.random.rand(size[n], size[n+1])*2 - 1
        tmp_weights = tmp_weights * weight_range
        arr.append(tmp_weights)
    weights = gnp.garray(arr)
    return arr


def gen_biases() -> array:
    arr = []
    global biases
    for n in range(1, len(size)):
        tmp_biases = np.random.rand(size[n])*2 - 1
        tmp_biases = tmp_biases * bias_range
        arr.append([tmp_biases for _ in range(input_dim[-1])])
    biases = gnp.garray(arr)
    return arr


def calc(a):
    out = a
    for layer in range(len(size) - 1):
        out = out * weights[layer]
        out = out + biases[layer]
        out = activation_func_vec(out)
    return out


def set_inputDim(i):
    global input_dim
    input_dim = i


def setSize(s):
    global size
    size = s
