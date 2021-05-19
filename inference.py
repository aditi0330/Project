import requests
import json
import os
import cv2
import sys
import numpy as np

classes = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}

def decode_predictions(predictions):
    labels = []
    for preds in predictions:
        labels.append(classes[np.argmax(preds)])

    return labels

def construct_image_batch(image_group, BATCH_SIZE):
    # get the max image shape
    max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

    # construct an image batch object
    image_batch = np.zeros((BATCH_SIZE,) + max_shape, dtype='float32')

    # copy all images to the upper left part of the image batch object
    for image_index, image in enumerate(image_group):
        image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

    return image_batch

def find_type(image):
    
    image_batch = construct_image_batch(image_group, len(image_group))
    predictions = make_serving_request(image_batch)
    labels = decode_predictions(predictions)

    return labels
    

def plot_model_history(model_name, history, epochs):
  
  print(model_name)
  plt.figure(figsize=(15, 5))
  
  # summarize history for accuracy
  plt.subplot(1, 2 ,1)
  plt.plot(np.arange(0, len(history['acc'])), history['acc'], 'r')
  plt.plot(np.arange(1, len(history['val_accuracy'])+1), history['val_accuracy'], 'g')
  plt.xticks(np.arange(0, epochs+1, epochs/10))
  plt.title('Trained from Scratch')
  plt.xlabel('')
  plt.ylabel('')
  plt.legend(['accuracy', 'val_accuracy'], loc='best')
  
  plt.subplot(1, 2, 2)
  plt.plot(np.arange(1, len(history['loss'])+1), history['loss'], 'r')
  plt.plot(np.arange(1, len(history['val_loss'])+1), history['val_loss'], 'g')
  plt.xticks(np.arange(0, epochs+1, epochs/10))
  plt.title('Trained from Scratch')
  plt.xlabel('')
  plt.ylabel('')
  plt.legend(['loss', 'val_loss'], loc='best')
  
  
  plt.show()

 plot_model_history('FCN_model', FCN_model_info.history, epochs)
