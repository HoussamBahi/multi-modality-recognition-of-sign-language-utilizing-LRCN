
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
#x_face,x_gait,  y = get_dataset(num_people)






face_input = tf.keras.layers.Input(shape=(32, 32, 3), name='face_input')


def extraction_rgb():
    num_people = 5
    x_face, x_gait, y = get_dataset(num_people)

    print("the shape of RGB is ", x_face.shape)


    xface = x_face[:, :1, :, :, :].reshape(45, 32, 32, 3)
    #print(x_face.shape)
    xgait = x_gait[:, :1, :, :, :].reshape(45, 32, 32, 3)
    # face_imput= tf.keras.layers.Input(shape = (32, 32, 3), name = 'face_imput')
    face_input = tf.keras.layers.Input(shape=(32, 32, 3), name='face_input')
    face_encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(face_input)
    face_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        face_encoder)
    face_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(face_encoder)
    face_encoder = tf.keras.layers.Flatten()(face_encoder)
    face_encoder = tf.keras.layers.Dense(units=4096, activation="relu")(face_encoder)
    face_encoder = tf.keras.layers.Dense(units=4096, activation="relu")(face_encoder)
    face_sendtogru = tf.keras.layers.Dense(32, activation='relu', name='face_dense_1')(face_encoder)

    x_face = x_face[:, :1, :, :, :1].reshape(45, 32, 32)
    #print(x_face.shape)
    x_gait = x_gait[:, :1, :, :, :1].reshape(45, 32, 32)

    face_input = tf.keras.layers.Input(shape=(32, 32), name='face_input')
    face_encoder = tf.keras.layers.GRU(16, return_sequences=True)(face_input)
    face_encoder = tf.keras.layers.Dropout(0.4, name='face_dropout_1')(face_encoder)
    face_encoder = tf.keras.layers.Flatten(name='face_flatten_1')(face_encoder)
    face_encoder = tf.keras.layers.Dense(64, activation='relu', name='face_dense_1')(face_encoder)
    face_encoder = tf.keras.layers.Dropout(0.4, name='face_dropout_2')(face_encoder)
    face_encoder = tf.keras.layers.BatchNormalization()(face_encoder)
    print("RGB Extraction shape",face_encoder.shape)
    return face_encoder

gait_input = tf.keras.layers.Input(shape=(32, 32, 3), name='gait_input')

def extraction_skeleton():
    num_people = 5
    x_face, x_gait, y = get_dataset(num_people)

    print("the shape of SKELETON is ", x_gait.shape)


    xface = x_face[:, :1, :, :, :].reshape(45, 32, 32, 3)
    #print(x_face.shape)
    xgait = x_gait[:, :1, :, :, :].reshape(45, 32, 32, 3)
    gait_input = tf.keras.layers.Input(shape=(32, 32, 3), name='gait_input')

    gait_encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_input)
    gait_encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(gait_encoder)
    gait_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(gait_encoder)
    gait_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(gait_encoder)
    gait_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        gait_encoder)
    gait_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(gait_encoder)
    gait_encoder = tf.keras.layers.Flatten()(gait_encoder)
    gait_encoder = tf.keras.layers.Dense(units=4096, activation="relu")(gait_encoder)
    gait_encoder = tf.keras.layers.Dense(units=4096, activation="relu")(gait_encoder)

    gait_sendtogru = tf.keras.layers.Dense(32, activation='relu', name='face_dense_1')(gait_encoder)

    gait_input = tf.keras.layers.Input(shape=(32, 32), name='gait_input')
    gait_encoder = tf.keras.layers.GRU(16, return_sequences=True)(gait_input)
    gait_encoder = tf.keras.layers.Dropout(0.4, name='gait_dropout_1')(gait_encoder)
    gait_encoder = tf.keras.layers.Flatten(name='gait_flatten_1')(gait_encoder)
    gait_encoder = tf.keras.layers.Dense(64, activation='relu', name='gait_dense_1')(gait_encoder)
    gait_encoder = tf.keras.layers.Dropout(0.4, name='gait_dropout_4')(gait_encoder)
    gait_encoder = tf.keras.layers.BatchNormalization()(gait_encoder)
    print("SKELETON Extraction shape",gait_encoder.shape)

    print("Output shape", y.shape)
    return gait_encoder

