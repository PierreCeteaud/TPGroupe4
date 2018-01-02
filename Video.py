# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:47:18 2017

@author: Pierre
"""

import cv2
import numpy as np
import math
fps=25

def Train_Video(Sequences,EcartFenetres,TailleFenetre,hz,center=True,cadree=False):
    # EcartFenetres et TailleFenetre sont donnés en secondes
    # Retourne une liste des features par fenetre
    # une ligne par fenêtre
    # une colonne par feature, la dernière colonne indique si la fenêtre est avec ou sans la présentatrice    
    # le nombre de fenêtres suit la règle définie par librosa.sftf
    Retour=[]
    for CurrentSequence in Sequences :
        # Temps en s de la sequence
        LargeurSequence=CurrentSequence[6]
        if LargeurSequence>=TailleFenetre:
            # Nombre de fenêtres dans cette séquence 
            if center:
                # La formule a été trouvée empiriquement
                NbFenetres=math.ceil((LargeurSequence+1/hz)/EcartFenetres)
            else:
                # Celle-ci aussi
                NbFenetres=math.ceil((LargeurSequence-TailleFenetre+1/hz)/EcartFenetres)
            # Comme le nombre de fenêtre ne tombe pas pile poil pour les espacer de EcartFenetres
            # à l'intérieur de la séquence on recalcule EcartFenetre et on l'appelle EspacementFenetre
            # On n'a pas la garantie que sftf garde les fenêtre à l'intérieur de la séquence
            if NbFenetres==1:
                # Cas très spécifique où la séquence fait exactement la taille d'une fenêtre
                EspacementFenetres=0
            else:
                EspacementFenetres=(LargeurSequence-TailleFenetre)/(NbFenetres-1)
            for iFenetre in range (NbFenetres):
                ImageMilieu=int(fps*(CurrentSequence[0].total_seconds()+iFenetre*EspacementFenetres))
                img=GetImage(ImageMilieu,cadree)
                grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                m = np.mean(grayImage)
                v= np.var(grayImage)
                Retour.append([m,v,CurrentSequence[3]])        
    return Retour


def GetImage(NumImage,cadree=False):
    if cadree:
        Name=f"Images/cadree-{NumImage}.png"        
    else:
        Name=f"Images/06-11-22-{NumImage}.png"
    return cv2.imread(Name)
