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
parser.add_argument("-strain","--save_train", type=str, default="save_encode/", help="Save dir of encoding of train images")
parser.add_argument("-smodel","--save_model", type=str, default="model/", help="Save dir of encoding of train images")
args = parser.parse_args()

'''Get save dir from args'''

save_dir = args.save_train
save_model = args.save_model

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

'''Create and train the SVC classifier '''

svm_clf = svm.SVC(gamma='scale', C=1, probability=True)
svm_clf.fit(encodings,names)

'''save the model to disk'''

filename = save_model + 'svm_model.sav'
pickle.dump(svm_clf, open(filename, 'wb'))

'''Print if done'''

print("Done")
