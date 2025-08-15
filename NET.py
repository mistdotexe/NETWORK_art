from array import array
from math import exp
import numpy as np   
from PIL import Image

def activation_func(x):
        return[(abs(k)+k)/2 for k in x]

class NeuralNetwork:
    def __init__(self):
        self.weights = [1]
        self.biases = [0]
        self.weight_range = [-1, 1]
        self.bias_range = [-1, 1]
        self.input_dim = 1
        self.size = (1, 1)
        self.imgBase = []


    def gen_weights(self) -> array:
        arr = []
        for n in range(len(self.size)-1):
            tmp_weights = np.random.rand(self.size[n], self.size[n+1])
            tmp_weights = tmp_weights * (self.weight_range[1]-self.weight_range[0]) + self.weight_range[0]
            arr.append(tmp_weights)
        self.weights = arr
        return arr


    def gen_biases(self) -> array:
        arr = []
        global biases
        for n in range(1, len(self.size)):
            tmp_biases = np.random.random(size = self.size[n])
            tmp_biases = tmp_biases * (self.bias_range[1]-self.bias_range[0]) + self.bias_range[0]
            arr.append(tmp_biases)
        self.biases = arr
        return arr


    def calc(self,a) -> array:
        out = np.array(a, dtype=np.float32)
        for layer in range(len(self.size) - 1):
            out = np.matmul(out, self.weights[layer])
            out = np.add(out, self.biases[layer])
            out = activation_func(out)
        return out


    def gen_image_base(self, width, height, zoom ):
        pixels = []
        for y in range(int(-height/2), int(height/2)):
            tmpx = []
            for x in range(int(-width/2), int(width/2)):
                tmpx.append([x/width*zoom, y/height*zoom*width/height])
            pixels.append(tmpx)
        self.imgBase = pixels
        return pixels
        
    def gen_image(self):
        img = self.calc(self.imgBase)
        max_val = np.max(img)
        img = img / max_val * 255
        img = np.uint8(img)
        img = Image.fromarray(img, mode = 'RGB')  
        return img