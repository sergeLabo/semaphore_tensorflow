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
Lit les images dans training_shot_resized
Convertit en une ligne
Convertit le gris de 0 à 1
Ajoute la lettre de l'image
Met tout dans train et test
Compresse

Variable:
train = 60000
test = 10000
size = 40
"""


import os
import numpy as np
import cv2

from pymultilame import MyTools


CHARS_DICT = {  "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7,
                "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
                "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
                "v": 21, "w": 22, "x": 23, "y": 24, "z": 25, " ": 26 }


def get_chars_label(img_file_name):
    """img_file_name = ... /semaphore_ia/shots/shot_000/shot_0_a.png
    Retourne le a
    remplace les chars anormaux en " "
    """
    l = img_file_name[-5]
    if l not in CHARS_DICT.keys():
        l = " "
    c = CHARS_DICT[l]
    return c


class ShotsCompression:

    def __init__(self, root, train, test, size, gray=0, imshow=1):
        """ root = dossier semaphore
            train = nombre d'images d'apprentissage
            test = nombre d'images de test
            size = taille des images pour l'ia
            gray = 0 ou 1 NB ou gray
            imshow = 0 ou 1 pour affichage d'image ou non pendant l'exécution
        """

        self.train = train
        self.test  = test
        self.size = size
        self.gray = gray
        self.imshow = imshow

        # Mon objet avec mes outils perso
        self.mytools = MyTools()

        # Valable avec exec ici ou en import
        self.root = root
        print("Dossier semaphore", self.root)

        self.get_images_list()
        a = "Nombre d'images: total = {}, apprentissage = {}, test = {}"
        print(a.format(len(self.images_list), train, test))

        if not self.gray:
            # entier 0 et 1
            self.images = np.zeros((self.train, self.size*self.size), dtype=np.uint8)
            self.images_test = np.zeros((self.test, self.size*self.size), dtype=np.uint8)
        else:
            # 0 à 1 en float
            self.images = np.zeros((self.train, self.size*self.size), dtype=float)
            self.images_test = np.zeros((self.test, self.size*self.size), dtype=float)

        # Entier
        self.labels = np.zeros((self.train), dtype=np.uint8)
        self.labels_test = np.zeros((self.test), dtype=np.uint8)

    def get_images_list(self):
        """Liste de toutes les images dans training_shot_resized avec leurs
        chemin absolu.
        """

        a = os.path.join(self.root, 'training_shot_resized')
        print("Dossier des images:", a)

        self.images_list = self.mytools.get_all_files_list(a, ".png")
        print("Nombre d'images =", len(self.images_list))

    def create_semaphore_npz(self):
        """Lit toutes les images de
        /media/data/3D/projets/semaphore/semaphore_ia/shots
        60 000 images 40x40
        transformation du array 40,40 en array 1, 2500
        conversion 0:255 en 0:1
        x_train = images = 60000x2500
        y_train = labels = 60000x1
        x_test = images = 10000x2500
        y_test = labels = 10000x1
        concatenate dans un gros array
        enregistrement
        """

        i = 0
        if self.imshow:
            cv2.namedWindow('Image')
        for f in self.images_list:
            # Lecture de l'image f
            img = cv2.imread(f, 0)

            if self.imshow:
                if i % 1000 == 0:
                    #print(i)
                    imgB = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)
                    cv2.imshow('Image', imgB)
                    cv2.waitKey(1)

            # Conversion du gris 0 à 255 en 0 à 1
            # img = np.reshape() / 255.0
            img = np.true_divide(img, 255)

            # Conversion en 0 et 1, soit noir et blanc, pas de gris
            # Sert à rien, c'est enregistré en int !!
            img = img.astype(int)

            # Retaillage sur une ligne
            img = np.resize(img, (self.size * self.size))

            # Labels
            label = get_chars_label(f)

            # Insertion par lignes
            if i < self.train:
                self.images[i] = img
                self.labels[i] =  label
            else:
                self.images_test[i - self.train] = img
                self.labels_test[i - self.train] =  label
            i += 1

        cv2.destroyAllWindows()
        self.save_npz()

    def save_npz(self):
        """Enregistre les arrays images et labels dans un fichier compressé
        ./semaphore.npz
        x_train = images = 60000x2500
        y_train = labels = 60000x1
        """

        outfile = os.path.join(self.root, 'semaphore.npz')
        np.savez_compressed(outfile, **{"x_train": self.images,
                             "y_train": self.labels,
                             "x_test":  self.images_test,
                             "y_test":  self.labels_test})

        print('Fichier compressé =', outfile)


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

    train, test, size, gray, imshow = 60000, 10000, 40, 0, 1

    # Compression des images
    sc = ShotsCompression(root, train, test, size, gray, imshow)
    sc.create_semaphore_npz()
