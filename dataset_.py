

import PySimpleGUI as sg
import os.path
import os



import os

import numpy
from keras.applications.vgg16 import VGG16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from IPython.display import display
from keras.utils.vis_utils import plot_model

def get_dataset(num_people):
  x1 = []
  x2 = []
  y  = []
  for i in range(num_people):
      ii = i - 1

      skpath = "E://code_swessen//code_mini//"
      rgb = skpath + "sr" + str(ii + 1) + "//"
      gait_folder = skpath + "sr" + str(ii + 1) + "//" + "skeleton" + "//"

      # gait_folder = 'processed_dataset' + '/' + 'person' + str(i+1) + '/' + 'gait' + '/'
      styles = os.listdir(gait_folder)
      styles.sort()
      #print("styles", styles)
      for j, style in enumerate(styles):
          style_folder = gait_folder + style + '/'
          angles = os.listdir(style_folder)
          angles.sort()
          #print("angles", angles)
          angles = angles[: 33]
          angles.sort()
          xx = []

          for image in angles:
              path = style_folder + image
              img = cv2.imread(path, -1)
              img = cv2.resize(img, (32, 32))
              img = np.array(img, dtype=np.float16)
              xx.append(img)
          xx = np.array(xx, dtype=np.float16)
          # x.append(xx)
          x = np.stack(xx, axis=3)
          x1.append(xx)




      skpath = "E://code_swessen//code_mini//"
      rgb = skpath + "sr" + str(ii + 1) + "//"
      gait_folder = skpath + "sr" + str(ii + 1) + "//" + "rgb" + "//"


      #gait_folder = 'processed_dataset' + '/' + 'person' + str(i+1) + '/' + 'gait' + '/'
      styles = os.listdir(gait_folder)
      styles.sort()
      #print("styles",styles)
      for j, style in enumerate(styles):
          style_folder = gait_folder + style + '/'
          angles = os.listdir(style_folder)
          angles.sort()
          #print("angles",angles)
          angles = angles[: 33]
          angles.sort()
          xx = []

          for image in angles:
                  path = style_folder + image
                  img = cv2.imread(path, -1)
                  img = cv2.resize(img, (32, 32))
                  img = np.array(img, dtype=np.float16)
                  xx.append(img)
                  #print("img",img.shape)
          xx = np.array(xx, dtype=np.float16)
          ##print("xx",xx.shape)
          # x.append(xx)
          x = np.stack(xx, axis=3)
          x2.append(xx)
      for j in range(9):
          y.append(i)
  x1 = np.asarray(x1, dtype = np.float32)
  x2 = np.array(x2, dtype = np.float64)
  y = tf.keras.utils.to_categorical(y)
  #x1,x2, y = shuffle(x1,x2, y)
  print("-----DATA EXTRACTED ... ... ... -----")
  return x1,x2, y

print("---------------------------------------")


num_people = 5
x_face,x_gait,  y = get_dataset(num_people)