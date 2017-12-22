# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:00:09 2017

@author: Pierre
"""

import cv2
import matplotlib.pyplot as plt
import os



cap = cv2.VideoCapture('06-11-22.mp4')
compresion_params=[cv2.IMWRITE_PNG_COMPRESSION,9];    
NumImage=0
while(cap.isOpened()):
    ret, frame = cap.read()
    NumImage+=1
    #plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    # Permet de reprendre l'enregistrement là où on était
    Name=f"06-11-22-{NumImage}.png"
    if not os.path.isfile(Name):
        cv2.imwrite(Name,frame, compresion_params)    
cap.release()
