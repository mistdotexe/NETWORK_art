import NET
from PIL import Image
import numpy as np
import random
import string
height = 1080
width = 1920
zoom = 2

tmp = []
for y in range(int(-height/2),int(height/2)):
    tmpx = []
    for x in range(int(-width/2),int(width/2)):
        tmpx.append([x/width*zoom,y/height*zoom*height/width])
    tmp.append(tmpx)
del tmpx
pixels = []
input_dim = height,width
size = 2,12,12,12,12,12,3
NET.set_inputDim(input_dim)
NET.setSize(size)
NET.gen_biases()
NET.gen_weights()

pixels = NET.calc(tmp)
print(np.shape(pixels))
del tmp
pixels = np.asarray(pixels)
pixels = 1/(np.exp(-pixels)+1)
pixels = pixels * 255
pixels = np.uint8(pixels)

new_image = Image.fromarray(pixels, mode = 'RGB')  
new_image.show()
new_image = new_image.convert('RGB') # if mode is HSV
name = str(width)+','+str(height)+'_'.join(random.choices(string.ascii_lowercase, k = 8))+'.png'
new_image.save(name)