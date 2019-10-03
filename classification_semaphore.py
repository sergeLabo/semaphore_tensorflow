#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import numpy as np
from tensorflow import keras

"""
Exemple construit sur documentation tensorflow:
    https://www.tensorflow.org/tutorials/keras/classification

    En utilisant les datas de:
        https://github.com/sergeLabo/semaphore

    Dans le fichier semaphore.npz:
        60 000 images training
        10 000 images test
        Les images font 40x40 pixels en gris, les valeurs de pixels 0 à 255
        sont déjà ramenées entre 0 et 1.
"""

# Chargement des images du semaphore
fichier = np.load('./semaphore.npz')

x_train = fichier['x_train']
y_train = fichier['y_train']
x_test = fichier['x_test']
y_test = fichier['y_test']

# 70 000 images de 40x40
train_images = x_train.reshape(60000, 40, 40)
train_labels = y_train
test_images = x_test.reshape(10000, 40, 40)
test_labels = y_test

L = "abcdefghijklmnopqrstuvwxyz "
# Liste des 27 noms d'objets dans le dataset semaphore
class_names = list(L)

"""
Build the model:
    Building the neural network requires configuring the layers of the model,
    then compiling the model.

    Set up the layers:
        The basic building block of a neural network is the *layer*. Layers
        extract representations from the data fed into them. Hopefully, these
        representations are meaningful for the problem at hand.

        Most of deep learning consists of chaining together simple layers. Most
        layers, such as `tf.keras.layers.Dense`, have parameters that are
        learned during training.

        The first layer in this network, `tf.keras.layers.Flatten`, transforms
        the format of the images
        from a two-dimensional array (of 40 by 40 pixels)
        to a one-dimensional array (of 40 * 40 = 1600 pixels).
        Think of this layer as unstacking rows of pixels in the image and
        lining them up. This layer has no parameters to learn; it only
        reformats the data.

        After the pixels are flattened, the network consists of a sequence
        of two `tf.keras.layers.Dense` layers. These are densely connected,
        or fully connected, neural layers. The first `Dense` layer has 128
        nodes (or neurons).

        The second (and last) layer is a 27-node *softmax* layer that returns
        an array of 27 probability scores that sum to 1. Each node contains
        a score that indicates the probability that the current image belongs
        to one of the 27 classes.
"""

model = keras.Sequential([  keras.layers.Flatten(input_shape=(40, 40)),
                            keras.layers.Dense(128, activation='relu'),
                            keras.layers.Dense(27, activation='softmax') ])

"""
Compile the model:
    Before the model is ready for training, it needs a few more settings.
    These are added during the model's *compile* step:

        * *Optimizer*
            This is how the model is updated based on the data it sees and its
            loss function.

        * *Loss function*
            This measures how accurate the model is during training.
            You want to minimize this function to "steer" the model in the
            right direction.

        * *Metrics*
            Used to monitor the training and testing steps. The following
            example uses *accuracy*, the fraction of the images that are
            correctly classified.
"""

model.compile(  optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'] )

"""
Training the model:
    Training the neural network model requires the following steps:

        1. Feed the training data to the model. In this example, the training
        data is in the `train_images` and `train_labels` arrays.
        2. The model learns to associate images and labels.
        3. You ask the model to make predictions about a test set—in this
        example, the `test_images` array. Verify that the predictions match the
        labels from the `test_labels` array.

    To start training, call the `model.fit` method—so called because it "fits"
    the model to the training data:
"""
model.fit(train_images, train_labels, epochs=2)

"""
As the model trains, the loss and accuracy metrics are displayed. This model
reaches an accuracy of about 0.88 (or 88%) on the training data.

Evaluate accuracy
    Next, compare how the model performs on the test dataset:
"""
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\nTesting ......")
print('    Efficacité sur les images test:', test_acc)

"""
It turns out that the accuracy on the test dataset is a little less than the
accuracy on the training dataset. This gap between training accuracy and test
accuracy represents *overfitting*. Overfitting is when a machine learning model
performs worse on new, previously unseen inputs than on the training data.

Make predictions
    With the model trained, you can use it to make predictions about some images.
"""
predictions = model.predict(test_images)

"""
Here, the model has predicted the label for each image in the testing set.
Let's take a look at the first prediction:
"""
print("Label de la 1ère image de test:", test_labels[0])

"""
print("predictions[0]", predictions[0])
A prediction is an array of 27 numbers. They represent the model's "confidence"
that the image corresponds to each of the 27 different objects.
You can see which label has the highest confidence value:
"""

print("Prédiction de la 1ère image:", np.argmax(predictions[0]))

"""
Finally, use the trained model to make a prediction about a single image.

`model.predict` returns a list of lists—one list for each image in the batch of data. Grab the predictions for our (only) image in the batch

# `tf.keras` models are optimized to make predictions on a *batch*, or collection, of examples at once. Accordingly, even though you're using a single image, you need to add it to a list

"""

print("\nTest sur 10 images")
for i in range(10):
    img = test_images[10*i]
    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # Now predict the correct label for this image
    # Retourne une liste de 27 prédictions
    predictions_single = model.predict(img)
    print("Image: {} Prédiction {}".format( test_labels[10*i],
                                            np.argmax(predictions_single[0])))
