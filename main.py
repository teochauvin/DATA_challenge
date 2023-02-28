##############################################
#        AUTOENCODER AND PCA APPROACHES      #
##############################################

"""
Author: tfsp 
Version : 1_AE (first then VAE) 

Comment: 
-  
- Performances sur le test_set : (en attente)

"""

## IMPORTS
from models.AE import Autoencoder

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import PIL
import PIL.Image
import pathlib

import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

## DATASET
batch_size = 32
img_height = 128
img_width = 128

# On charge les datasets train et validation (80/20)
train_ds = tf.keras.utils.image_dataset_from_directory(
  "dataset/train",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "dataset/train",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

## LOAD MODEL 

ae = Autoencoder(latent_dim=2)

## TRAINING
ae.train(train_ds, val_ds, 1, batch_size)

## SHOW
ae.bottleneckMap(train_ds)

## SAVE 
ae.save_ED("model_ae_rep_1.h5")


