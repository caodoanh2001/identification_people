import face_recognition
from sklearn import svm, neighbors
import os
import numpy as np
from numpy import save
from numpy import load
import pickle
from tqdm import tqdm, notebook
from os import path
from bounding_box import bounding_box as bb
import cv2
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("-i","--img_dir", type=str, default="save_encode/", help="image dir")
parser.add_argument("-smodel","--save_model", type=str, default="model/", help="Save dir of model")
args = parser.parse_args()

'''Get model'''

img_dir = args.img_dir
save_model = args.save_model + 'svm_model.sav'

'''Define model'''

model = pickle.load(open(save_model, 'rb'))

'''Test img '''

test_image = face_recognition.load_image_file(img_dir)

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = model.predict([test_image_enc])
    print(*name)

x_min = face_locations[0][2]
y_min = face_locations[0][0]
x_max = face_locations[0][1]
y_max = face_locations[0][3]

color_list = ["maroon", "green", "yellow", "purple", "fuchsia", "lime", "red", "silver"]
id_color = random.randint(0, 7)
box_color = color_list[id_color]
bb.add(test_image, x_min, y_min, x_max, y_max, str(name[0]), box_color)

cv2.imshow('demo', test_image)
cv2.waitKey(0) 
