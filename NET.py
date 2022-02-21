from array import array
from math import exp
import cupy as cp


weights = []
biases = []
weight_range = 4
bias_range = 1
input_dim = 1
size = 1, 1


def activation_func(x):
    return cp.tanh(x)
activation_func_vec = cp.vectorize(activation_func)


def gen_weights() -> array:
    arr = []
    global weights
    for n in range(len(size)-1):
        tmp_weights = cp.random.random(size = (size[n], size[n+1]), dtype=cp.float32)*2 - 1
        tmp_weights = tmp_weights * weight_range
        arr.append(tmp_weights)
    weights = arr
    return arr


def gen_biases() -> array:
    arr = []
    global biases
    for n in range(1, len(size)):
        tmp_biases = cp.random.random(size = size[n], dtype=cp.float32)*2 - 1
        tmp_biases = tmp_biases * bias_range
        arr.append(tmp_biases)
    biases = arr
    return arr


def calc(a):
    out = cp.array(a, dtype=cp.float32)
    for layer in range(len(size) - 1):
        out = cp.matmul(out, weights[layer])
        out = cp.add(out, biases[layer])
        out = activation_func_vec(out)
    return out


def set_inputDim(i):
    global input_dim
    input_dim = i


def setSize(s):
    global size
    size = s
