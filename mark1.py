
# imports that we need
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



#this function is responsible for getting the dataset for both rgb and skeleton
def get_dataset(num_people):

    # three arrays x1 x2 y for rgb skeleton and output respectively
  x1 = []
  x2 = []
  y  = []

    # first loop for the ppl that we chose ( in this example we have 5 ppl )
  for i in range(num_people):
      ii = i - 1


      # path for rgb

      skpath = "E://code_swessen//code_mini//"

      face_folder = skpath + "sr" + str(ii + 1) + "//" + "rgb" + "//"

      styles = os.listdir(face_folder)
      styles.sort()
      # print("styles",styles)

      # loop for every style in each ppl

      for j, style in enumerate(styles):
          style_folder = face_folder + style + '/'
          angles = os.listdir(style_folder)
          angles.sort()
          # print("angles",angles)

          #here we take 33 images in every style
          # angles means here the sequence or group of images

          angles = angles[: 33]
          angles.sort()
          xx = []

          #loop for getting each image in every style
          for image in angles:
                  path = style_folder + image
                  img = cv2.imread(path, -1)
                  img = cv2.resize(img, (32, 32))
                  img = np.array(img, dtype=np.float16)
                  xx.append(img)
                  #print("img",img.shape)
          xx = np.array(xx, dtype=np.float16)
          # print("xx",xx.shape)
          # x.append(xx)
          x = np.stack(xx, axis=3)

          # here x1 contain each image for each style for each ppl
          x1.append(xx)

      # path for skeleton

      skpath = "E://code_swessen//code_mini//"

      gait_folder = skpath + "sr" + str(ii + 1) + "//" + "skeleton" + "//"


      styles = os.listdir(gait_folder)
      styles.sort()
      # print("styles", styles)

      # loop for every style in each ppl
      for j, style in enumerate(styles):
          style_folder = gait_folder + style + '/'
          angles = os.listdir(style_folder)
          angles.sort()
          # print("angles", angles)
          angles = angles[: 33]
          angles.sort()
          xx = []

          # loop for getting each image in every style
          for image in angles:
              path = style_folder + image
              img = cv2.imread(path, -1)
              img = cv2.resize(img, (32, 32))
              img = np.array(img, dtype=np.float16)
              xx.append(img)
          xx = np.array(xx, dtype=np.float16)
          # x.append(xx)
          x = np.stack(xx, axis=3)

          # here x2 contain each image for each style for each ppl
          x2.append(xx)



      # here we define the output ( 9*5 = 45) 9 styles for each 5 ppl
      for j in range(9):
          y.append(i)
  x1 = np.asarray(x1, dtype = np.float32)
  x2 = np.array(x2, dtype = np.float64)
  y = tf.keras.utils.to_categorical(y)
  #x1,x2, y = shuffle(x1,x2, y)
  return x1,x2, y


#x1, x2, y = shuffle(x1, x2, y)
# xx1 = np.asarray(x1).astype(tf.float16)
# xx2 = np.array(x2).astype(tf.float16)

# setting that we have 5 ppl

num_people = 5

# calling dataset function

x_face,x_gait,  y = get_dataset(num_people)
#print(x_face,y)

# print(x_face.shape)
# print("gait",x_gait.shape)
# print(y.shape)
# print(xx1.shape)
# print(xx2.shape)


#
# tt= x_face
# ttt=tt
# n=len(ttt[-1])
#
# for i in range(n):
#     tt[1]=ttt[-1]
# tt[2]=ttt[1]
# tt[3]=ttt[2]
# tt[4]=ttt[3]
# print("tt",tt.shape)


#tt = np.array(tt, dtype=np.float32)
#print("tt: ",tt.shape)

# print('x_face:', x_face.shape)
#
# print('y     :', y.shape)

# reshaping the arrays to feed the VGG16 model

xface=x_face[:,:1,:,:,:].reshape(45,32,32,3)
# print(x_face.shape)
xgait=x_gait[:,:1,:,:,:].reshape(45,32,32,3)




# structure VGG16 RGB

#face_imput= tf.keras.layers.Input(shape = (32, 32, 3), name = 'face_imput')
face_input = tf.keras.layers.Input(shape = (32, 32, 3), name = 'face_input')
face_encoder= tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(face_input)
face_encoder= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(face_encoder)
face_encoder= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(face_encoder)
face_encoder=tf.keras.layers.Flatten()(face_encoder)
face_encoder=tf.keras.layers.Dense(units=4096,activation="relu")(face_encoder)
face_encoder=tf.keras.layers.Dense(units=4096,activation="relu")(face_encoder)
face_sendtogru= tf.keras.layers.Dense(32, activation = 'relu', name = 'face_dense_1')(face_encoder)



# structure GRU RGB
# reshaping the arrays to feed the GRU model
x_face=x_face[:,:1,:,:,:1].reshape(45,32,32)
# print(x_face.shape)
x_gait=x_gait[:,:1,:,:,:1].reshape(45,32,32)



face_input = tf.keras.layers.Input(shape = (32, 32), name = 'face_input')
face_encoder = tf.keras.layers.GRU(16, return_sequences=True)(face_input)
face_encoder = tf.keras.layers.Dropout(0.4, name = 'face_dropout_1')(face_encoder)
face_encoder = tf.keras.layers.Flatten(name = 'face_flatten_1')(face_encoder)
face_encoder = tf.keras.layers.Dense(64, activation = 'relu', name = 'face_dense_1')(face_encoder)
face_encoder = tf.keras.layers.Dropout(0.4, name = 'face_dropout_2')(face_encoder)
face_encoder = tf.keras.layers.BatchNormalization()(face_encoder)




# structure VGG16 Skeleton

gait_input = tf.keras.layers.Input(shape = (32, 32, 3), name = 'gait_input')

gait_encoder= tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(gait_input)
gait_encoder= tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(gait_encoder)
gait_encoder= tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(gait_encoder)
gait_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(gait_encoder)
gait_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(gait_encoder)
gait_encoder= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(gait_encoder)
gait_encoder=tf.keras.layers.Flatten()(gait_encoder)
gait_encoder=tf.keras.layers.Dense(units=4096,activation="relu")(gait_encoder)
gait_encoder=tf.keras.layers.Dense(units=4096,activation="relu")(gait_encoder)

gait_sendtogru= tf.keras.layers.Dense(32, activation = 'relu', name = 'face_dense_1')(gait_encoder)

# structure GRU Skeleton


gait_input = tf.keras.layers.Input(shape = (32, 32), name = 'gait_input')
gait_encoder = tf.keras.layers.GRU(16,  return_sequences=True)(gait_input)
gait_encoder = tf.keras.layers.Dropout(0.4, name = 'gait_dropout_1')(gait_encoder)
gait_encoder = tf.keras.layers.Flatten(name = 'gait_flatten_1')(gait_encoder)
gait_encoder = tf.keras.layers.Dense(64, activation = 'relu', name = 'gait_dense_1')(gait_encoder)
gait_encoder = tf.keras.layers.Dropout(0.4, name = 'gait_dropout_4')(gait_encoder)
gait_encoder = tf.keras.layers.BatchNormalization()(gait_encoder)


#
# def weighted_average(tensors):
#   alpha = 0.5
#   face = tensors[0]
#   gait = tensors[1]
#   weighted_average = alpha * face + (1 - alpha) * gait
#   # print(face)
#   # print(gait)
#   return weighted_average
# # print(face_encoder.shape)
# # print(gait_encoder.shape)
# # print(face_encoder)
# # weig = np.concatenate((face_encoder, gait_encoder), axis=0)
# # print("weig",weig.shape)
# decoder = tf.keras.layers.Lambda(weighted_average, name = 'face_gait_weighted_average')([face_encoder, gait_encoder])
# # print(decoder.shape)



# face_input = tf.keras.layers.Input(shape = (4096, 1), name = 'face_input')
#
# decoder = tf.keras.layers.GRU(16, return_sequences=True)(face_input)
# decoder = tf.keras.layers.Dropout(0.4, name = 'face_dropout_1')(decoder)
# decoder = tf.keras.layers.Flatten(name = 'face_flatten_1')(decoder)
# decoder = tf.keras.layers.Dense(64, activation = 'relu', name = 'face_dense_1')(decoder)
# decoder = tf.keras.layers.Dropout(0.4, name = 'face_dropout_2')(decoder)
# decoder = tf.keras.layers.BatchNormalization()(decoder)

#"""""""""""""""""""""""""""""""""""""""""" RBG model
# last layer ( couche) in our RGB model1
decoder1 = tf.keras.layers.Dense(5, activation = 'softmax', name = 'output')(face_encoder)

model1 = tf.keras.models.Model(
      inputs = [face_input],
      outputs = decoder1)
# compiling the model1
model1.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])


# showing the model1 info
model1.summary()


#"""""""""""""""""""""""""""""""""""""""""" Skeleton model


# last layer ( couche) in our skeleton model2

decoder2 = tf.keras.layers.Dense(5, activation = 'softmax', name = 'output')(gait_encoder)

model2 = tf.keras.models.Model(
      inputs = [gait_input],
      outputs = decoder2)

# compiling the model2
model2.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model2.summary()


#
# plot = plot_model(model, show_shapes = True, to_file = 'model.png', show_layer_names = True)
# display(plot)

# setting batch size and epochs and validation size ( 20% in our ex)
EPOCHS = 10
BATCH_SIZE = 1
VALID_SPLIT = 0.2

acc_train1 = []
acc_val1 = []
acc_train2 = []
acc_val2 = []
print("----------------RGB Training--------------------------")
history1 = model1.fit([x_face], y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = VALID_SPLIT)
acc_train1 += history1.history['accuracy']
acc_val1 += history1.history['val_accuracy']

print("----------------SKELETON Training--------------------------")

history2 = model2.fit([x_gait], y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = VALID_SPLIT)
acc_train2 += history2.history['accuracy']
acc_val2 += history2.history['val_accuracy']


#
# history = model.fit([x_face, x_gait], y, batch_size = BATCH_SIZE, epochs = 10, validation_split = VALID_SPLIT)
# acc_train += history.history['accuracy']
# acc_val += history.history['val_accuracy']

print("The model is being evaluated")

#evaluate our model to calculate the accuracy and the loss evaluer notre modele pour calculer le taux et la perte

# test_loss, test_acc = model.evaluate([x_face, x_gait], verbose=0)
# print("The accuracy of the model is:")
#
# # print accuracy and loss
# print(test_acc)
# print(test_loss)

# graph to show accuracy

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 18

plt.plot(acc_train1, linewidth=2)
plt.plot(acc_val1, linewidth=2)
plt.plot(acc_train2, linewidth=2)
plt.plot(acc_val2, linewidth=2)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train rgb', 'Test rgb','Train skeleton','Test skeleton'], loc='upper left')
plt.grid()

plt.show()


# graph to show accuracy with loss
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['font.size'] = 14

fig, ax = plt.subplots(1, 2)

ax[0].plot(history1.history['accuracy'])
ax[0].plot(history1.history['val_accuracy'])
ax[0].plot(history2.history['accuracy'])
ax[0].plot(history2.history['val_accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train rgb', 'Test rgb','train skeleton','test skeleton'], loc='upper left')
ax[0].grid()

ax[1].plot(history1.history['loss'])
ax[1].plot(history1.history['val_loss'])
ax[1].plot(history2.history['loss'])
ax[1].plot(history2.history['val_loss'])

ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train rgb', 'Test rgb','train skeleton','test skeleton'], loc='upper left')
ax[1].grid()

plt.show()


# last accuracy obtained

print("Training RGB accuracy    ",acc_train1[-1])
print("Validation RGB accuracy  ",acc_val1[-1])

print("Training SKELETON accuracy    ",acc_train2[-1])
print("Validation SKELETON accuracy  ",acc_val2[-1])

