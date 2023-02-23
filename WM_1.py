##############################################
#             FIRST NAIVE MODLE              #
##############################################

"""
Author: tfsp 
Version : 1_CONVOLUTIONNAL 

Comment: 
- Modèle très simple et naif : 
- Performances sur le test_set : F1-score : 0.94

"""

## IMPORTS ## 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras import layers

import matplotlib.pyplot as plt

## GLOBALS ##

## DATASETS ## 

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

# DEBUG : On affiche des exemples 
class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])

    plt.axis("off")

plt.show()

## MODEL ##

num_classes = 2

# Modèle sequentiel avec différentes couches de convolutions pour limiter le nombre de paramètres 
# Normalization + SOFTMAX en sortie 
# Le pooling permet de donner des degrés de liberté
model = tf.keras.Sequential([
  layers.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activaion="softmax")  
])

# On optimise avec ADAM 
# Fonction de perte : categorical crossEntropy (problème de classification) 
# Metrique accuracy adaptee pour de la classification supervisée 
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# On fit le modèle, 1 epoch semble suffisante mais sur une plus grosse machine on peut aller plus loin
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1
)

# On sauvegarde le modèle pour ne pas itérer trop souvent ces étapes gourmandes en calcul 
model.save("modelv1_save.h5")
