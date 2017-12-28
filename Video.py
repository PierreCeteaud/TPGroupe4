# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:47:18 2017

@author: Pierre
"""

import cv2
import numpy as np
import math
fps=25

def Train_Video(Sequences,EcartFenetres,TailleFenetre):
    # EcartFenetres et TailleFenetre sont donnés en secondes
    # Retourne une liste des features par fenetre
    # une ligne par fenêtre
    # une colonne par feature, la dernière colonne indique si la fenêtre est avec ou sans la présentatrice
    # Sur chaque séquence l'écart est ajusté légèrement à la baisse pour faire entrer un nombre entier de fenêtres
    Retour=[]
    for CurrentSequence in Sequences :
        # Temps en s de la sequence
        LargeurSequence=CurrentSequence[6]
        if LargeurSequence>=TailleFenetre:
            # Nombre de fenêtres dans cette séquence arrondi au dessus
            NbSequences=math.ceil(1+(LargeurSequence-TailleFenetre)/EcartFenetres)
            # Comme on a arrondi au dessus on vas prendre un EcartFenetres un peu plus petit
            # que demandé et on l'appelle EspacementFenetre
            if NbSequences==1:
                # Cas très spécifique où la séquence fait exactement la taille d'une fenêtre
                EspacementFenetres=0
            else:
                EspacementFenetres=(LargeurSequence-TailleFenetre)/(NbSequences-1)
            for iFenetre in range (NbSequences):
                ImageMilieu=int(fps*(CurrentSequence[0].total_seconds()+iFenetre*EspacementFenetres))
                img=GetImage(ImageMilieu)
                grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                m = np.mean(grayImage)
                v= np.var(grayImage)
                Retour.append([m,v,CurrentSequence[3]])        
    return Retour

def GetImage(NumImage):
    Name=f"Images/06-11-22-{NumImage}.png"
    return cv2.imread(Name)
