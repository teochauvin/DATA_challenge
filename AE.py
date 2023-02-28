import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers

class Autoencoder(keras.Model): 

    def __init__(self, latent_dim): 
        super(Autoencoder, self).__init__() 

        self.tag = "AE"

        self.input_size = 128
        self.channel = 3
        self.batch_size = 32

        """ # encoder NON CONVOLUTIONAL
        inputs  = keras.Input(shape=(self.input_size, self.input_size, self.channel))
        x       = layers.Flatten()(inputs)
        x       = layers.Dense(1024, activation='relu')(x)
        x       = layers.Dense(512, activation='relu')(x)
        x       = layers.Dense(128, activation='relu')(x)
        x       = layers.Dense(32, activation='relu')(x)
        z       = layers.Dense(latent_dim)(x) 

        self.encoder = keras.Model(inputs, z, name="encoder")
        self.encoder.summary()

        # decoder NON CONVOLUTIONAL
        inputs_latent  = keras.Input(shape=(latent_dim,))
        x       = layers.Dense(32, activation='relu')(inputs_latent)
        x       = layers.Dense(128, activation='relu')(x)
        x       = layers.Dense(512, activation='relu')(x)
        x       = layers.Dense(1024, activation='relu')(x)
        x       = layers.Dense(49152, activation='sigmoid')(x)
        outputs = layers.Reshape(target_shape=(self.input_size, self.input_size, self.channel), input_shape=(49152,))(x)

        self.decoder = keras.Model(inputs_latent, outputs, name="decoder")
        self.decoder.summary()""" 

        # encoder 
        inputs    = keras.Input(shape=(self.input_size, self.input_size, 3))
        x         = layers.Conv2D(32, 3, strides=1, padding="same", activation="relu")(inputs)
        x         = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x         = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x         = layers.Conv2D(64, 3, strides=1, padding="same", activation="relu")(x)
        x         = layers.Flatten()(x)
        outputs   = layers.Dense(latent_dim, activation="relu")(x)

        self.encoder = keras.Model(inputs,outputs, name="encoder")
        self.encoder.summary()

        # decoder 
        inputs_latent  = keras.Input(shape=(latent_dim,))
        x       = layers.Dense(32 * 32 * 64, activation="relu")(inputs_latent)
        x       = layers.Reshape((32, 32, 64))(x)
        x       = layers.Conv2DTranspose(64, 3, strides=1, padding="same", activation="relu")(x)
        x       = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x       = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        outputs = layers.Conv2DTranspose(3,  3, padding="same", activation="sigmoid")(x)

        self.decoder = keras.Model(inputs_latent, outputs, name="decoder")
        self.decoder.summary()


    def train_step(self, input):
        '''
        Implementation of the training update.
        Receive an input, compute loss, get gradient, update weights and return metrics.
        Here, our metrics are loss.
        args:
            inputs : Model inputs
        return:
            loss    : Total loss
            r_loss  : Reconstruction loss
            kl_loss : KL loss
        '''
        
        # ---- Get the input we need, specified in the .fit()
        #
        if isinstance(input, tuple):
            input = input[0]
        
        # ---- Forward pass
        #      Run the forward pass and record 
        #      operations on the GradientTape.
        #
        with tf.GradientTape() as tape:
            
            # ---- Get encoder & decoder outputs
            #
            z              = self.encoder(input)
            reconstruction = self.decoder(z)
         
            # ---- Compute loss
            #      Reconstruction loss, KL loss and Total loss
            #
            loss  = tf.reduce_mean(keras.losses.binary_crossentropy(input, reconstruction))

            

        # ---- Retrieve gradients from gradient_tape
        #      and run one step of gradient descent
        #      to optimize trainable weights
        #
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {"loss": loss}

    def predict(self, inputs): 

        z = self.encoder.predict(inputs)
        outputs = self.decoder.predict(z)
        return outputs 

    def call(self, inputs):
        
        z       = self.encoder(inputs)
        output  = self.decoder(z)
        return output

    def train(self, train_images, test_images, numEpochs, batchSize): 
        
        print(" Compiling AE models ...\n")
        self.encoder.compile()
        self.decoder.compile() 
        self.compile(optimizer='adam')

        print("Fiting AE ... \n")
        history = self.fit(train_images, epochs=numEpochs, batch_size=batchSize, shuffle=True)
        return history

    def bottleneckMap(self, input): 

        z = self.encoder.predict(input) 

        plt.figure(figsize=(14, 10))
        plt.scatter(z[:, 0] , z[:, 1], alpha=0.5, s=30)
        plt.title("AE latent space")
        plt.colorbar()
        plt.savefig(f"performances/ae_latentspace_representation.png")
        plt.show()

    def save_ED(self,filename):
        '''Save model in 2 part'''

        self.encoder.save(f'save/AE-{filename}-encoder.h5')
        self.decoder.save(f'save/AE-{filename}-decoder.h5')

        print('AE Saved.')

    def reload_ED(self,filename):
        '''Reload a 2 part saved model.'''

        self.encoder = keras.models.load_model(f'save/AE-{filename}-encoder.h5')
        self.decoder = keras.models.load_model(f'save/AE-{filename}-decoder.h5')

        print('AE Reloaded.')

    def reconstruction_loss(self, inputs): 
        """
            Binary crossentropy function but similar detection with MSE
        """

        reconstruction = self.predict(inputs) 
        return tf.reduce_mean(keras.losses.binary_crossentropy(inputs, reconstruction), [1,2]).numpy()

        
        