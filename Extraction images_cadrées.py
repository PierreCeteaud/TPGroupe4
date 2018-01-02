# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:00:09 2017

@author: Pierre
"""

import cv2
import os

cap = cv2.VideoCapture('06-11-22.mp4')
compresion_params=[cv2.IMWRITE_PNG_COMPRESSION,9];    
NumImage=0
while(cap.isOpened()):
    ret, frame = cap.read()
    NumImage+=1
    # Le while ne sort jamais de la boucle, on le teste ici
    if frame is None:
        break
    # on suprime le bandeau qui commence au 200ème pixel
    # on supprime les 15 premiers pixel en hauteur
    # et les 40 pixels à gauche et à droite
    # car les personnes sont bien cadrées au centre
    # (sauf pour les pals à plusieurs personnes)
    subframe=frame[15:200,40:-40]    
    # Permet de reprendre l'enregistrement là où on était
    Name=f"Images/cadree-{NumImage}.png"
    if not os.path.isfile(Name):
        cv2.imwrite(Name,subframe, compresion_params)    
          
cap.release()


