# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:14:37 2017

@author: Pierre
"""

import numpy as np
import librosa
import librosa.display
import librosa.output
import matplotlib.pyplot as plt

Signal,hz= librosa.load('06-11-22.wav')
len(Signal)/hz
Fenetre=Signal[90*hz:(91)*hz]

librosa.display.waveplot(Fenetre, sr=hz)

X = librosa.feature.mfcc(Fenetre,hz,n_mfcc=24)

plt.figure(figsize=(10, 4))
librosa.display.specshow(X, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

# Ca fonctionne, mais on a rien fait

np.shape(X) # 24 features avec 44 valeurs espacées dans le temps. Pourquoi 44 ?, je pensais trouver un vecteur pas une matrice
X[1,1] # C'est un tableau de complexes