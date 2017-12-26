# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:14:37 2017

@author: Pierre
"""

import numpy as np
import librosa
import librosa.display
import librosa.output
import librosa.feature as feature
import matplotlib.pyplot as plt
import scipy

Signal,hz= librosa.load('06-11-22.wav')
len(Signal)/hz



Sequence=Signal[int(234.6*hz):int(237.97*hz)]


# Calcul exact du "hop" à partir du nombre de valeur qu'on veut obtenir
#Nb_val=44
#hop_l=int(np.size(Fenetre)/(Nb_val-1))

# 1 valeur en sortie du mfcc =1 fenêtre
Taille_Fenetre=1 # seconde
win_l=hz*Taille_Fenetre
hop_l=int(win_l/2)
win_l=int(win_l)

mfcc = librosa.feature.mfcc(Sequence,
                            hz,
                            n_mfcc=13,
                            n_mels=24, 
                            hop_length=hop_l)
print(np.shape(mfcc)) 


## à partir d'un code trouvé sur stackoveflow

window = 'hamming'
fmin = 20
fmax = hz/2

#np.abs permet d'extraire la norme du signal ("magnitude")
#pourquoi **2 ???
D = np.abs(librosa.stft(Sequence, 
                        window=window, 
                        n_fft=win_l, 
                        win_length=win_l, 
                        hop_length=hop_l))**2

# calcul du MEL
S = feature.melspectrogram(S=D, y=Sequence, n_mels=24, fmin=fmin, fmax=fmax)
# calcul des 13 coefficients
feats = feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)

plt.figure(figsize=(10, 4))
librosa.display.specshow(feats, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()


plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

# Calcul de la dérivée
mfcc_delta = librosa.feature.delta(mfcc)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_delta, x_axis='time')
plt.colorbar()
plt.title('MFCC_delta')
plt.tight_layout()

# Calcul de la dérivée seconde
mfcc_delta2 = librosa.feature.delta(mfcc_delta)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_delta, x_axis='time')
plt.colorbar()
plt.title('MFCC_delta²')
plt.tight_layout()


print(np.shape(mfcc)) 
np.shape(mfcc_delta)
# Ca fonctionne, mais on n'a rien fait

np.shape(X) # 24 features avec 44 valeurs espacées dans le temps. Pourquoi 44 ?, je pensais trouver un vecteur pas une matrice
np.size(Sequence)

X[1,1] # C'est un tableau de complexes

print(librosa.__version__)