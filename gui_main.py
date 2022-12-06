


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

    print("the shape of extracted RGB is ", x_face.shape)


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

    print("the shape of extracted SKELETON is ", x_gait.shape)
    print("Output shape", y.shape)

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
    return gait_encoder



def weighted_average(tensors):
  alpha = 0.5
  face = tensors[0]
  gait = tensors[1]
  weighted_average = alpha * face + (1 - alpha) * gait
  #print(face)
  #print(gait)
  return weighted_average


def fusion_rgb_skeleton():
    face_encoder = extraction_rgb()
    gait_encoder = extraction_skeleton()

    #print(face_encoder.shape)
    #print(gait_encoder.shape)
    print("---------------------------------------------")
    print("Fusion vector info",face_encoder)

    decoder = tf.keras.layers.Lambda(weighted_average, name='face_gait_weighted_average')([face_encoder, gait_encoder])
    print("Fusion vector shape",decoder.shape)

    decoder = tf.keras.layers.Dense(5, activation='softmax', name='output')(decoder)

    model = tf.keras.models.Model(
        inputs=[face_input, gait_input],
        outputs=decoder)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model





# The GUI CODE

file_list_column = [
    [
    sg.Text("Image Folder"),
    sg.In(size=(15,1), enable_events=True, key="-FOLDER-"),
    sg.FolderBrowse(),
   ],
   [
     sg.Listbox(
         values=[], enable_events=True, size= (30,20),key="-FILE LIST-"
     )
   ],

]

image_viewer_column=[
    [sg.Text("Choose a process from the list below : ")],
    [sg.Button("GET DATASET")],
    [sg.Button("EXTRACTION")],
    [sg.Button("FUSION")],
    [sg.Button("TRAIN")],
    [sg.Text("path: ")],
    [sg.Text(size=(40,1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Text("Training accuracy: ")],
    [sg.Text(size=(40,1), key="-train-")],
    [sg.Text("Validation accuracy: ")],
    [sg.Text(size=(40,1), key="-validation-")],
]

layout = [
    [
    sg.Column(file_list_column),
    sg.VSeparator(pad=(20,0)),
    sg.Column(image_viewer_column),
        ]
]
window = sg.Window("ACTION EXPRESSION RECOGNITION", layout)




while True:
    event, values = window.read()
    if event=="GET DATASET":
        try:
            get_dataset(5)

        except:

            pass

    if event=="EXTRACTION":
        try:
            extraction_rgb()
            extraction_skeleton()

        except:
            pass

    if event=="FUSION":
        try:
            fusion_rgb_skeleton()

        except:
            pass
    if event=="TRAIN":
        try:
            import rgb_import
            print("---------------------- training completed -------------------------")
            #train_model()
            #os.system('python rgb_import.py')
            acct = rgb_import.acc_train
            accv = rgb_import.acc_val
            #print(acct[-1], accv[-1])
            t = window["-train-"].update(acct[-1])
            v = window["-validation-"].update(accv[-1])





        except:
            pass
    if event == "Exit" or event== sg.WIN_CLOSED:
        break
    if event=="-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list=[]
        fnames = [
            f
          for f in file_list
          if os.path.isfile(os.path.join(folder, f))
          and f.lower().endswith((".jpg",".png", ".gif"))

        ]
        window["-FILE LIST-"].update(fnames)
    elif event== "-FILE LIST-":
        try:
            filename= os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]

            )
            print(filename)
            pathimg= filename
            #testimage(pathimg)
            pt=window["-TOUT-"].update(filename)
            #window["-IMAGE-"].update(filename=filename)



        except:
            pass






