# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:00:09 2017

@author: Pierre
"""

import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('06-11-22.mp4')

# Attention on affiche toutes les images du film
while(cap.isOpened()):
    ret, frame = cap.read()
    plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
