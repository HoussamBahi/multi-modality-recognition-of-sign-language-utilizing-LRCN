import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

trdata = ImageDataGenerator()

traindata = trdata.flow_from_directory(directory="E://code_swessen//code_mini//sr0//rgb",target_size=(224,224))

num_people=5
tsdata = ImageDataGenerator()
for i in range(num_people):
    ii = i - 1

    skpath = "E://code_swessen//code_mini//"
    rgb = skpath + "sr" + str(ii + 1) + "//"
    skelton_folder = skpath + "sr" + str(ii + 1) + "//" + "skeleton" + "//"

    testdata = tsdata.flow_from_directory(directory=skelton_folder, target_size=(224,224))

print(testdata.filepaths)
