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
parser.add_argument("-test","--test_dir", type=str, default="save_encode_test/", help="Save dir of encoding of train images")
parser.add_argument("-smodel","--save_model", type=str, default="model/", help="Save dir of encoding of train images")
args = parser.parse_args()

'''Get save dir from args'''

save_dir = args.test_dir
save_model = args.save_model + 'svm_model.sav'

'''Define list variable to store encoding'''

encodings = []
names = []

'''Load encoding'''

persons = os.listdir(save_dir)
for i in tqdm(range(0, len(persons))):
  npys = os.listdir(save_dir + persons[i])
  for npy in npys:
    encodings.append(np.load(save_dir + persons[i] + '/' + npy))
    names.append(persons[i])

'''Define model'''

model = pickle.load(open(save_model, 'rb'))

'''Test and get accuracy'''

predict_names = []
for faces in encodings:
  predict_name = model.predict([faces])
  predict_names.append(*predict_name)

check = [predict == real for predict, real in zip(predict_names, names)]
print('accuracy: ', sum(check)/len(predict_names) * 100)
