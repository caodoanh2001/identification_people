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

parser = argparse.ArgumentParser()
parser.add_argument("-train","--train_dir", type=str, default="data/", help="Train directory")
parser.add_argument("-test","--test_dir", type=str, default="test/", help="Test directory")
parser.add_argument("-strain","--save_train", type=str, default="save_encode/", help="Save dir of encoding of train images")
parser.add_argument("-stest","--save_test", type=str, default="save_encode_test/", help="Save dir of encoding of test images")
parser.add_argument("-type","--type_img", type=str, default=".jpg", help="Type img")
parser.add_argument("-ltrain","--load_train", type=int, default=1, help="Type img")
parser.add_argument("-ltest","--load_test", type=int, default=1, help="Type img")

args = parser.parse_args()

'''Define some directory variable from args'''

train_dir = args.train_dir
test_dir = args.test_dir
save_train_dir = args.save_train
save_test_dir = args.save_test
type_img = args.type_img

data_folder = [train_dir, test_dir]
save_directory = [save_train_dir, save_test_dir]

train_list = os.listdir('data/')
test_list = os.listdir('test/')
load_data = [train_list, test_list]
save_train = os.listdir(save_train_dir)
load_train_test = [args.load_train , args.load_test]

'''Encoding images'''

for folder, save_dir, load, data_dir in zip(load_data, save_directory, load_train_test, data_folder):
  print('Loading ... ', data_dir)
  if (load):
    for i in tqdm(range(0, len(folder))):
        pix = os.listdir(data_dir + folder[i])
        # Loop through each training image for the current person
        
        if not path.exists(save_dir):
          os.mkdir(save_dir)
       
        person_dir = save_dir + folder[i]
        
        if not path.exists(person_dir):
          os.mkdir(person_dir)
        
        for person_img in pix:

            if (not path.exists(person_dir + '/' + person_img.split(type_img)[0] + '.npy')):
                # Get the face encodings for the face in each image file
                face = face_recognition.load_image_file(data_dir + folder[i] + "/" + person_img)
                face_bounding_boxes = face_recognition.face_locations(face)

                #If training image contains exactly one face
                if len(face_bounding_boxes) == 1:
                    face_enc = face_recognition.face_encodings(face)[0]
                    # Add face encoding for current image with corresponding label (name) to the training data
                    #encodings.append(face_enc)
                    save(person_dir + '/' + person_img.split(type_img)[0] + '.npy', face_enc)
                    #names.append(folder[i])
                # else:
                #     print(person + "/" + person_img + " was skipped and can't be used for training")

