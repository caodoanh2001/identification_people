import face_recognition
from sklearn import svm, neighbors
import os
import numpy as np
from numpy import save
from numpy import load
import pickle
from tqdm import tqdm, notebook
from os import path
import cv2
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("-test","--test_dir", type=str, default="save_encode_test/", help="Save dir of test images")
parser.add_argument("-smodel","--save_model", type=str, default="model/", help="Save dir of model")
args = parser.parse_args()

'''Get save dir from args'''

test_dir = args.test_dir
save_model = args.save_model + 'svm_model.sav'

'''Define list variable to store encoding'''

test_list = os.listdir(test_dir)

'''Define model'''

model = pickle.load(open(save_model, 'rb'))

'''Test and get accuracy'''

predict_names = []
name_img = []
real_names = []
percentages = []
for i in tqdm(range(0, len(test_list))):
  img_list = os.listdir(test_dir + test_list[i])

  for img in img_list:
    result = 'unknown'
    test_image = face_recognition.load_image_file(test_dir + test_list[i] + '/' + img)

    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    if (no == 1):
      test_image_enc = face_recognition.face_encodings(test_image)[0]
      name = model.predict([test_image_enc])
      if (max(model.predict_proba([test_image_enc])[0]) > 0.3):
        result = name[0]
    
    predict_names.append(result)
    name_img.append(img)
    real_names.append(test_list[i])
    if max(model.predict_proba([test_image_enc])[0]) > 0.3:
      percentages.append(max(model.predict_proba([test_image_enc])[0]))
    else:
      percentages.append(0)

      
for name, result, real, percentage in zip(name_img, predict_names, real_names, percentages):
  print('Image: {} - Predict: {} - Real: {} - Percentages: {}%'.format(name, result, real, round(percentage,3)*100))

check = [predict == real for predict, real in zip(predict_names, real_names)]
print('accuracy: ', sum(check)/len(predict_names) * 100)
print('Predict actually: {}/{}'.format(sum(check),len(predict_names)))
