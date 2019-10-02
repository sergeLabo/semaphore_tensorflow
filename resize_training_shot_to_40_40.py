#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Retaille les images de training_shot à 40x40

    Renomme training_shot en training_shot_copy

    Crée training_shot

    Lit les images de training_shot_copy
    Les retaille
    Les enregistre dans training_shot
"""


import os
import numpy as np
import cv2

from pymultilame import MyTools


class ResizeTrainingShot:


    def __init__(self, root, size):
        self.root = root  # soit ..../semaphore
        self.size = int(size)

        # Mes outils personnels
        self.tools = MyTools()

        # Renommage de training_shot en training_shot_copy
        self.rename_training_shot()

        # Re-création des dossiers
        self.create_training_shot_dir()
        self.create_sub_folders()

        # Liste des images
        self.shot_list = self.get_shot_list()

    def rename_training_shot(self):
        ori = os.path.join(self.root, "training_shot")
        dest = os.path.join(self.root, "training_shot_copy")
        os.rename(ori, dest)

    def create_training_shot_dir(self):
        directory = os.path.join(self.root, "training_shot")
        print("Dossier training_shot:", directory)
        self.tools.create_directory(directory)

    def create_sub_folders(self):
        """Création de n dossiers shot_000"""

        # Nombre de dossiers nécessaires
        d = os.path.join(self.root, "training_shot_copy")
        n = len(self.tools.get_all_sub_directories(d)) -1
        print("Nombre de sous répertoires", n)
        for i in range(n):
            directory = os.path.join(self.root, 'training_shot',
                                                'shot_' + str(i).zfill(3))
            self.tools.create_directory(directory)
        print("Sous répertoires créés")

    def get_shot_list(self):
        """Liste des images"""

        # Liste
        shot = os.path.join(self.root, "training_shot_copy")
        shot_list = self.tools.get_all_files_list(shot, ".png")

        print("Dossier des images NB:", shot)
        print("Nombre d'images:", len(shot_list))

        return shot_list

    def change_resolution(self, img, x, y):
        """Une image peut-être ratée"""

        try:
            res = cv2.resize(img, (x, y), interpolation=cv2.INTER_AREA)
        except:
            res = np.zeros([self.size, self.size, 1], dtype=np.uint8)
        return res

    def get_new_name(self, shot):

        return shot.replace("/training_shot_copy/", "/training_shot/")

    def create_training_shot_resized_dir(self):

        directory = os.path.join(self.root, "training_shot_resized")
        print("Dossier training_shot_resized:", directory)
        self.tools.create_directory(directory)

    def batch(self):
        """Lecture, resize, save"""

        i = 0

        # Pour chaque image
        for shot in self.shot_list:
            # Lecture
            img = cv2.imread(shot, 0)

            # Resize
            img_out = self.change_resolution(img, self.size, self.size)
            i += 1

            # Save
            new_shot = self.get_new_name(shot)
            print(new_shot)
            cv2.imwrite(new_shot, img_out)



if __name__ == "__main__":

    SIZE = 40

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

    print("\nResize de toutes les images de  training_shot")

    rts = ResizeTrainingShot(root, SIZE)
    rts.batch()
    print("Done")
