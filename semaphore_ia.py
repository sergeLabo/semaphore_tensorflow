#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

########################################################################
# This file is part of Semaphore.
#
# Semaphore is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Semaphore is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
########################################################################

"""
Training: Calcule le fichier 'd'intelligence' weight.npy avec 60 000 images
Testing: Teste avec 10 000 images
Le PEP 8 80 colonnes n'est pas respecté pour donner la priorité aux explications
"""


import os
import shutil
import numpy as np
import cv2
from pymultilame import MyTools


def sigmoid(x):
    """La fonction sigmoïde est une courbe en S:
    https://fr.wikipedia.org/wiki/Sigmo%C3%AFde_(math%C3%A9matiques)
    """

    return 1 / (1 + np.exp(-x))

def sigmoid_prime(z):
    """La dérivée de la fonction sigmoid,
    soit sigmoid' comme f' !
    """

    return z * (1 - z)

def relu(x):
    """Rectifie les négatifs à 0:
    -1 > 0
     1 > 1
     Rectified Linear Unit:

    In the context of artificial neural networks, the rectifier is an
    activation function defined as the positive part of its argument.
    https://bit.ly/2HyT4ZO sur Wikipedia en.
     """

    return np.maximum(0, x)

def relu_prime(z):
    """Fonction: 1 pour tous les réels positifs ou nuls et 0 pour les réels négatifs.

    La fonction de Heaviside (également fonction échelon unité, fonction
    marche d'escalier) est la fonction indicatrice de R.
    Une fonction fonction indicatrice, est une fonction définie sur un
    ensemble E qui explicite l’appartenance ou non à un sous-ensemble F de E
    de tout élément de E.
    """

    return np.asarray(z > 0, dtype=np.float32)


class SemaphoreIA:
    """Réseau de neuronnes Perceptron multicouches."""

    def __init__(self, root, train, learningrate, failed=0, imshow=1):
        """ root = dossier semaphore
            train = Nombre de shot pour l'apprentissage
            learningrate = coeff important
            failed = 0 ou 1, pour analyse des ratés
            imshow = 0 ou 1 pour affichage d'image ou non pendant l'exécution
        """

        self.root = root
        self.train = train
        self.learningrate = learningrate
        self.failed = failed
        self.imshow = imshow
        self.tools = MyTools()

        # Dossier des ratés
        if self.failed:
            # Suppression du dossier failed et recréation pour le vider
            try:
                shutil.rmtree(os.path.join(self.root, 'failed'))
                print("Suppression de failed")
            except:
                print('Pas de dossier failed')
            self.tools.create_directory(os.path.join(self.root, 'failed'))

        # Réseau de neurones: colonne 1600 en entrée, 2 nodes de 100, sortie de 27 caractères
        self.layers = [1600, 100, 100, 27]
        # Fonction d'activation: imite l'activation d'un neuronne
        self.activations = [relu, relu, sigmoid]

        fichier = np.load(os.path.join(self.root, 'semaphore.npz'))

        self.x_train = fichier['x_train']
        self.y_train = fichier['y_train']

        self.x_test = fichier['x_test']
        self.y_test = fichier['y_test']

        a = "Training: Shot {} Lettre {}; Testing: Shot {} Lettre {}"
        print(a.format( len(self.x_train), len(self.y_train),
                        len(self.x_test),  len(self.y_test)))

    def training(self):
        """Apprentissage avec 60 000 images
        Poids enregistré dans weights.npy
        """

        print("Training...")

        # Affichage des images pour distraire les mangalore
        if self.imshow:
            cv2.namedWindow('img')

        # Matrice diagonale de 1
        diagonale = np.eye(27, 27)

        # globals() Return a dictionary representing the current global symbol table.
        self.activations_prime = [globals()[fonction.__name__ + '_prime'] for fonction in self.activations]

        node_dict = {}

        # Liste des poids
        # Initialisation des poids des nodes, pour ne pas à être à 0
        # Construit 3 matrices (100x1600, 100x100, 27x100)
        # /np.sqrt() résultat expérimental de l'initialisation de Xavier Glorot et He
        weight_list = [np.random.randn(self.layers[k+1], self.layers[k]) / \
                       np.sqrt(self.layers[k]) for k in range(len(self.layers)-1)]

        # vecteur_ligne = image en ligne à la 1ère itération
        # nombre_lettre = nombre correspondant à la lettre de l'image
        # i pour itération, vecteur_colonne = x_train de i, nombre_lettre = y_train de i
        for i, (vecteur_ligne, nombre_lettre) in enumerate(zip(self.x_train, self.y_train)):

            # Affichage pour distraire les mangalore
            if self.imshow:
                if i % 400 == 0:
                    img = vecteur_ligne * 255
                    img = img.reshape(40,40)
                    img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)
                    cv2.imshow("img", img)
                    cv2.waitKey(1)

            # la ligne devient colonne
            vecteur_colonne = np.array(vecteur_ligne, ndmin=2).T

            # Forward propagation
            node_dict[0] = vecteur_colonne
            for k in range(len(self.layers)-1):
                # weight_list[k] (100x1600, 100x100 27x100) vecteur_colonne (1600,)
                # z de format 100 x 1
                z = np.dot(weight_list[k], vecteur_colonne)

                # self.activations = non linéaire sinon sortie fonction linéaire de l'entrée
                # imite le seuil d'activation électrique du neuronne
                vecteur_colonne = self.activations[k](z)

                node_dict[k+1] = vecteur_colonne

            # Retro propagation, delta_a = écart entre la sortie réelle et attendue
            delta_a = vecteur_colonne - diagonale[:,[nombre_lettre]]
            # Parcours des nodes en sens inverse pour corriger proportionnellemnt
            # les poids en fonction de l'erreur par rapport à la valeur souhaitée
            # Descente du Gradient stochastique
            for k in range(len(self.layers)-2, -1, -1):
                delta_z = delta_a * self.activations_prime[k](node_dict[k+1])
                delta_w = np.dot(delta_z, node_dict[k].T)
                delta_a = np.dot(weight_list[k].T, delta_z)
                # Pour converger vers le minimum d'erreur
                weight_list[k] -= self.learningrate * delta_w

        # Dans un fichier
        np.save(os.path.join(self.root, 'weights.npy'), weight_list)
        print('weights.npy enregistré')
        cv2.destroyAllWindows()
        return weight_list

    def testing(self):
        """Teste avec les images de testing, retourne le ratio de bon résultats"""

        print("Testing...")

        weight_list = np.load(os.path.join(self.root, 'weights.npy'))

        # Nombre de bonnes reconnaissance
        success = 0

        for vecteur_ligne, nombre_lettre in zip(self.x_test, self.y_test):
            # image en ligne au 1er passage pour les failed
            img = vecteur_ligne.copy()

            for k in range(len(self.layers)-1):
                vecteur_ligne = self.activations[k](np.dot(weight_list[k],
                                                      vecteur_ligne))

            reconnu = np.argmax(vecteur_ligne)
            if reconnu == nombre_lettre:
                success += 1
            else:
                if self.failed:
                    self.tools.create_directory(os.path.join(self.root,
                                                             'failed',
                                                             'bad_' + str(nombre_lettre)))
                    self.write_failed(img, nombre_lettre, reconnu, success)

        if len(self.x_test) != 0:
            resp = 100.0 * success / len(self.x_test)
        else:
            resp = 0
        return resp

    def write_failed(self, img, nombre_lettre, reconnu, S):
        """Les images avec erreur de reconnaisance sont copiées dans
        /semaphore/failed/bad_11/11_6_9067.png
        11 est la lettre k, donc dans le dossier il ny a que la lettre k
        et le 2ème nombre est la lettre reconnue fausse
        """
        name = str(nombre_lettre) + '_' + str(reconnu) + '_'  + str(S) + '.png'

        fichier = os.path.join( self.root,
                                'failed',
                                'bad_' + str(nombre_lettre),
                                name)

        img = img.reshape(40,40) * 255
        cv2.imwrite(fichier, img)


if __name__ == "__main__":

    # Chemin courrant
    abs_path = MyTools().get_absolute_path(__file__)
    print("Chemin courrant", abs_path)

    # Nom du script
    name = os.path.basename(abs_path)
    print("Nom de ce script:", name)

    # Abs path de semaphore sans / à la fin
    parts = abs_path.split("semaphore")
    root = os.path.join(parts[0], "semaphore")
    print("Path de semaphore:", root)

    train = 60000
    learningrate = 0.023
    sia = SemaphoreIA(root, train, learningrate, failed=1, imshow=1)
    sia.training()
    resp = sia.testing()
    print("Learningrate: {} Résultat {}".format(learningrate, round(resp, 1)))
