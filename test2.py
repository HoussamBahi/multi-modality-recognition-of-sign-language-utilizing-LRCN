

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
      print("styles", styles)
      for j, style in enumerate(styles):
          style_folder = gait_folder + style + '/'
          angles = os.listdir(style_folder)
          angles.sort()
          print("angles", angles)
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
      print("styles",styles)
      for j, style in enumerate(styles):
          style_folder = gait_folder + style + '/'
          angles = os.listdir(style_folder)
          angles.sort()
          print("angles",angles)
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
          print("xx",xx.shape)
          # x.append(xx)
          x = np.stack(xx, axis=3)
          x2.append(xx)
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
num_people = 5
x_face,x_gait,  y = get_dataset(num_people)
#print(x_face,y)

print(x_face.shape)
print("gait",x_gait.shape)
print(y.shape)
# print(xx1.shape)
# print(xx2.shape)



tt= x_face
ttt=tt
n=len(ttt[-1])

for i in range(n):
    tt[1]=ttt[-1]
tt[2]=ttt[1]
tt[3]=ttt[2]
tt[4]=ttt[3]
print("tt",tt.shape)


#tt = np.array(tt, dtype=np.float32)
#print("tt: ",tt.shape)

print('x_face:', x_face.shape)

print('y     :', y.shape)

x_face=x_face[:,:1,:,:,:].reshape(45,32,32,3)
print(x_face.shape)
x_gait=x_gait[:,:1,:,:,:].reshape(45,32,32,3)


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

# model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#g=tf.keras.layers.GRU

# base_model = VGG16(include_top=False, weights=None, input_tensor=None,
#             input_shape=( 32, 32,3))

#
# face_input = tf.keras.layers.Input(shape = (33, 32, 32, 3), name = 'face_input')
#
# face_encoder = tf.keras.layers.ConvLSTM2D(16, kernel_size = 3, activation = 'relu', name = 'face_conv_1')(face_input)
# face_encoder = tf.keras.layers.Dropout(0.4, name = 'face_dropout_1')(face_encoder)
# face_encoder = tf.keras.layers.Flatten(name = 'face_flatten_1')(face_encoder)
# face_encoder = tf.keras.layers.Dense(64, activation = 'relu', name = 'face_dense_1')(face_encoder)
# face_encoder = tf.keras.layers.Dropout(0.4, name = 'face_dropout_2')(face_encoder)
# face_encoder = tf.keras.layers.BatchNormalization()(face_encoder)

# gait_input = tf.keras.layers.Input(shape = (33, 32, 32, 3), name = 'gait_input')
#
# gait_encoder = tf.keras.layers.ConvLSTM2D(16, kernel_size = 3, name = 'gait_convlstm_1')(gait_input)
# gait_encoder = tf.keras.layers.Dropout(0.4, name = 'gait_dropout_1')(gait_encoder)
# gait_encoder = tf.keras.layers.Flatten(name = 'gait_flatten_1')(gait_encoder)
# gait_encoder = tf.keras.layers.Dense(64, activation = 'relu', name = 'gait_dense_1')(gait_encoder)
# gait_encoder = tf.keras.layers.Dropout(0.4, name = 'gait_dropout_4')(gait_encoder)
# gait_encoder = tf.keras.layers.BatchNormalization()(gait_encoder)



def weighted_average(tensors):
  alpha = 0.5
  face = tensors[0]
  gait = tensors[1]
  weighted_average = alpha * face + (1 - alpha) * gait
  return weighted_average
print(face_encoder.shape)
print(gait_encoder.shape)
decoder = tf.keras.layers.Lambda(weighted_average, name = 'face_gait_weighted_average')([face_encoder, gait_encoder])
print(decoder.shape)
decoder = tf.keras.layers.Dense(5, activation = 'softmax', name = 'output')(decoder)

model = tf.keras.models.Model(
      inputs = [face_input, gait_input],
      outputs = decoder)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#
# plot = plot_model(model, show_shapes = True, to_file = 'model.png', show_layer_names = True)
# display(plot)


EPOCHS = 10
BATCH_SIZE = 10
VALID_SPLIT = 0.3

acc_train = []
acc_val = []

history = model.fit([x_face, x_gait], y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = VALID_SPLIT)
acc_train += history.history['accuracy']
acc_val += history.history['val_accuracy']


#
# history = model.fit([x_face, x_gait], y, batch_size = BATCH_SIZE, epochs = 10, validation_split = VALID_SPLIT)
# acc_train += history.history['accuracy']
# acc_val += history.history['val_accuracy']

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 18

plt.plot(acc_train, linewidth=2)
plt.plot(acc_val, linewidth=2)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()

plt.show()



plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['font.size'] = 14

fig, ax = plt.subplots(1, 2)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Test'], loc='upper left')
ax[0].grid()

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Test'], loc='upper left')
ax[1].grid()

plt.show()


