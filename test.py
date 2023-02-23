##############################################
#     BUILD TEST DATASET, PREDICT, WRITE     #
##############################################

"""
Author: tfsp 
Version : 1_LAPTOP_build 

Comment: 
- fonction build : fabrique le dataset de test, converti chaque image en un array numpy, et les stocke dans des fichiers differents pour eviter l'overflow RAM
en effet 71000 images de 128x128x3 ou chaque couleur est codée sur 8bits : ~28GB qui dépasse largement les capacité d'un laptop 
- fonction get renvoie l'argmax pour faire la prédiction 
- make_pred : charge un a un les fichiers de test, réalise les prédictions, et écrit dans le csv les résultats 
A noter qu'encore une fois on peut facilement dépasser les capacités de RAM, 
d'ou l'utilisation explicite du garbage collector pour libérer de l'espace RAM au sein meme de l'execution du programme. 

Dans le cas général vous n'arriverez pas à build et prédire sur la même execution, à moins de rajouter manuellement l'utilisation du garbage collector. 

Moins vous avez de RAM, plus il vous faudra d'étape de calculs (augmenter le paramètre n)
"""

### IMPORTS 

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import PIL
import PIL.Image
import pathlib

import tensorflow as tf

import csv
import matplotlib.pyplot as plt
from random import randint

import gc

### FONCTIONS 

def get(predictions): 

    r = []

    for i in range(len(predictions)): 
        r.append(str(np.argmax(predictions[i]))) 

    return r

# build a dataset in different file to avoid RAM overflow during computation : (recommended n= 60/RAM(GB))  
def build_test_ds(n):

    d = 71504//n 

    test_ds = []
    with open('dataset/template_submissions.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')

        count=0

        for row in reader:

            if count>0:

                path = row[0]
                image = tf.keras.utils.load_img(f'dataset/test/{path}')
                input_arr = tf.keras.utils.img_to_array(image)
                test_ds.append(np.array(input_arr))

                if count%d==0:
                    print(count)
                    print(np.array(test_ds).shape)
                    np.save(f'ordered_test_ds_{count//d}.npy', np.array(test_ds), allow_pickle=True)
                    test_ds=[]
            
            count+=1 

    # le dernier save
    if len(test_ds)>0:
        print(count)
        np.save(f'ordered_test_ds_{n}.npy', np.array(test_ds), allow_pickle=True)

def display(csvfile): 

    with open('dataset/'+csvfile, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')


        tmp=[]

        for row in reader:
            tmp.append((row['Image'], 'eolienne' if row['Categorie'] == "1" else "background"))

    for j in range(len(tmp)//10):
        plt.figure(figsize=(30, 10))
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            k = randint(0,70000)
            plt.imshow(PIL.Image.open(pathlib.Path(f"dataset/test/{tmp[k][0]}")))
            plt.title(tmp[k][1])

            plt.axis("off")
        plt.show()

plt.show()

# realize prediction with a given trained model 
def make_pred(model_path, n): 

    model = tf.keras.models.load_model(model_path)
    pred = []
    start = 0

    # for each file of ordered images
    for j in range(1,n+1): 

        print("Loading next part of test dataset ... ")
        images = np.load(f'ordered_test_ds_{j}.npy',allow_pickle=True)

        print("computing predictions ... ")
        pred = get(model.predict(images))

        print("adding new predictions...")
        start = write_csv(start, pred)

        print("free RAM to avoid OOM Kill ... ")
        del images 
        del pred 
        gc.collect()
            
def write_csv(id_start, pred): 

    id_end = id_start+len(pred)
    print(f"Writing window ]{id_start}; {id_end}] ...")

    csvfile = 'template_submissions.csv'
    with open('dataset/'+csvfile, 'r') as fin, open('dataset/new'+csvfile, 'a') as fout:
        index = 0

        for line in iter(fin.readline, ''):

            if index<=id_start+len(pred) and index>id_start: 
                fout.write(line.replace('\n', pred[index-(id_start+1)] + '\n'))

            # if first line of all (headers)
            if index==0 and id_start==0: 
                fout.write(line)

            index += 1

    return id_start+len(pred)

### MAIN 

n = 4

#build_test_ds(n) 
#make_pred("modelv1_save.h5", 4) 

display('template_submissions_RAC.csv')