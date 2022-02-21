import NET
from PIL import Image
import cupy as cp
import numpy as np
import random
import string
width = 1920
height = 1080
zoom = 2

tmp = []
for y in range(int(-height/2),int(height/2)):
    tmpx = []
    for x in range(int(-width/2),int(width/2)):
        tmpx.append([x/width*zoom,y/height*zoom*height/width])
    tmp.append(tmpx)
del tmpx
pixels = []
size = 2,12,12,12,12,12,3
input_dim = height,width,size[0]
NET.set_inputDim(input_dim)
NET.setSize(size)
NET.gen_biases()
NET.gen_weights()

pixels = NET.calc(tmp)
print(cp.shape(pixels))
del tmp
pixels = 1/(cp.exp(-pixels)+1)
pixels = pixels * 255
pixels = cp.asnumpy(pixels)
pixels = np.uint8(pixels)

new_image = Image.fromarray(pixels, mode = 'RGB')  
new_image.show()
new_image = new_image.convert('RGB') # if mode is HSV
name = str(width)+','+ str(height)+'_'+''.join(random.choices(string.ascii_lowercase, k = 8))+'.png'
new_image.save(name)