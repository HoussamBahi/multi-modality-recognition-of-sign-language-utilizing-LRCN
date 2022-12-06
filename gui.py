
# imports that we need

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



# The GUI CODE

# left side of the GUI

file_list_column = [
    [
    sg.Text("Image Folder"),
    sg.In(size=(15,1), enable_events=True, key="-FOLDER-"),
    sg.FolderBrowse(),
   ],
   [
     sg.Listbox(
         values=[], enable_events=True, size= (30,25),key="-FILE LIST-"
     )
   ],

]

# right side of GUI

image_viewer_column=[
    [sg.Text("Choose a process from the list below : ")],
    [sg.Button("GET DATASET")],
    [sg.Button("EXTRACTION")],

    [sg.Button("TRAIN")],
    [sg.Text("path: ")],
    [sg.Text(size=(40,1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Text("Training RGB accuracy: ")],
    [sg.Text(size=(40,1), key="-trainR-")],
    [sg.Text("Validation RGB accuracy: ")],
    [sg.Text(size=(40,1), key="-validationR-")],
    [sg.Text("Training SKELETON accuracy: ")],
    [sg.Text(size=(40,1), key="-trainS-")],
    [sg.Text("Validation SKELETON accuracy: ")],
    [sg.Text(size=(40,1), key="-validationS-")],
]

# setting the general layout with splitting the two sides with a vertical line

layout = [
    [
    sg.Column(file_list_column),
    sg.VSeparator(pad=(20,0)),
    sg.Column(image_viewer_column),
        ]
]


window = sg.Window("ACTION EXPRESSION RECOGNITION", layout)



# loop to make our GUI functional
while True:

    # events for each action ( button ) and values to get the value
    event, values = window.read()
    if event=="GET DATASET":
        try:
            import dataset_
            #os.system('python dataset_.py')

        except:

            pass

    if event=="EXTRACTION":
        try:

            # extraction_rgb()
            # extraction_skeleton()
            import rgb_skeleton_extraction
            rgb_skeleton_extraction.extraction_rgb()
            rgb_skeleton_extraction.extraction_skeleton()

        except:
            pass


    if event=="TRAIN":
        try:
            import mark1
            print("---------------------- training completed -------------------------")
            #train_model()
            #os.system('python rgb_import.py')
            acct1 = mark1.acc_train1
            accv1 = mark1.acc_val1
            #print(acct[-1], accv[-1])
            t = window["-trainR-"].update(acct1[-1])
            v = window["-validationR-"].update(accv1[-1])

            acct2 = mark1.acc_train2
            accv2 = mark1.acc_val2
            # print(acct[-1], accv[-1])
            tt = window["-trainS-"].update(acct2[-1])
            vv = window["-validationS-"].update(accv2[-1])





        except:
            pass
    if event == "Exit" or event== sg.WIN_CLOSED:
        break

    # show images in the chosen folder
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
          and f.lower().endswith((".png", ".gif"))

        ]
        window["-FILE LIST-"].update(fnames)

    # show path when we choose an image

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








