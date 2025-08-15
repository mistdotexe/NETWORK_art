import NET
from PIL import Image
import numpy as np
import random
import string
import os 
import tkinter as tk
from PIL import ImageTk



# Function to choose a parent and generate children
def choose_parent(n):
    global parent
    global children
    global parent_image
    parent = children[n]
    parent_image = child_imges[n]
    print(f"Parent chosen: {n}")
    make_children()
    update_labels()


def make_children():
    global children
    global child_imges
    # can be made faster by not reinitializing the children
    children = [NET.NeuralNetwork() for _ in range(child_amount)]
    child_imges = []
    for child in children:
        child.imgBase = img_base
        child.weights = [w + np.random.normal(-0.05, 0.05, size=w.shape) for w in parent.weights]
        child.biases = [b + np.random.normal(-0.05, 0.05, size=b.shape) for b in parent.biases]
        child.size = size
        child.input_dim = input_dim
        child_imges.append(child.gen_image())
    print("Children generated")


def update_labels():
    # Convert parent_image to ImageTk.PhotoImage and update parent_label
    parent_image_tk = ImageTk.PhotoImage(parent_image)
    parent_label.config(image=parent_image_tk)
    parent_label.image = parent_image_tk  # Prevent garbage collection
    

    # Convert each child image to ImageTk.PhotoImage and update child_labels
    for i, child_label in enumerate(child_labels):
        child_image_tk = ImageTk.PhotoImage(child_imges[i])
        child_label.config(image=child_image_tk)
        child_label.image = child_image_tk  # Prevent garbage collection
        
    print("Labels updated")



width = 100
height = 100
zoom = 10
child_amount = 9

# Create Tkinter window
root = tk.Tk()
root.title("Neural Network Art")

# Create a list to hold parent and child images
child_imges = []
parent_image = None

#initialize parent and children labels
parent_label = tk.Label(root, text="Parent Neural Network")
child_labels = [tk.Label(root) for _ in range(child_amount)]


# Define the size of the neural network layers

size = 2,12,3
input_dim = height,width,size[0]

#create a parent neural network
parent = NET.NeuralNetwork()
parent.input_dim = input_dim
parent.size = size
parent.weight_range = [-1, 1]
parent.bias_range = [0, 21]
parent.gen_biases()
parent.gen_weights()

#generate image base and image
img_base = parent.gen_image_base(width, height, zoom)
imgArray = parent.gen_image()

# Makes this initial parent and children images
parent_image = parent.gen_image()  
make_children()


parent_label.bind("<Button-1>", lambda event: (make_children(), update_labels()))
for i, child_label in enumerate(child_labels):
    child_label.bind("<Button-1>", lambda event, index=i: choose_parent(index))
parent_label.pack()
for child_label in child_labels:
    child_label.pack(side=tk.LEFT, padx=5, pady=5)
update_labels()


# code to save the image
#name = str(width)+','+ str(height)+'_'+''.join(random.choices(string.ascii_lowercase, k = 8))+'.png'
#cwd = os.getcwd()
#parent_image.save(cwd +"\\images\\" + name)


#create labels to display the images
root.mainloop()
