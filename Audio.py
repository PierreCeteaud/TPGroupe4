# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:41:00 2017

@author: Pierre
"""

import numpy as np
import librosa
import librosa.display
import librosa.output
import librosa.feature as feature

window = 'hamming'
fmin = 20
# Cette limite est à étudier, en effet on trouvera facilement que la voix humaine
# varie entre 80 et 1500 hz
fmax = 4000
n_mfcc=24
try:
    print("On conserve le signal à",hz,"hz")
except:
    Signal,hz= librosa.load('06-11-22.wav')


def Train_Audio(Sequences,EcartFenetres,TailleFenetre,center=True):
    # EcartFenetres et TailleFenetre sont donnés en secondes
    # Retour_Xne une liste des features par fenetre
    # une ligne par fenêtre
    # une colonne par feature, la dernière colonne indique si la fenêtre est avec ou sans la présentatrice
    # le nombre de fenêtres suit la règle définie par librosa.sftf
    Retour_X=[]
    Retour_Y=[]
    Retour_NumSequence=[]
    win_l=hz*TailleFenetre
    hop_l=int(win_l*EcartFenetres)
    win_l=int(win_l)
    for i in range(len(Sequences)):
        CurrentSequence=Sequences[i]
        Sequence=Signal[int(CurrentSequence[0].total_seconds()*hz)
                        :int(CurrentSequence[1].total_seconds()*hz)]
        D = np.abs(librosa.stft(Sequence, 
                                window=window, 
                                n_fft=win_l, 
                                win_length=win_l, 
                                hop_length=hop_l,
                                center=center))**2
        # calcul du MEL
        S = feature.melspectrogram(S=D, y=Sequence, n_mels=24, fmin=fmin, fmax=fmax)
        # calcul des 13 coefficients
        mfcc = feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
        # Calcul de la dérivée
        mfcc_delta = librosa.feature.delta(mfcc)
        # Calcul de la dérivée seconde
        mfcc_delta2 = librosa.feature.delta(mfcc_delta)
        # Concatenation des features et Rajout de la ligne de présence
        Nb_Fenetres=mfcc.shape[1]
        f=np.vstack((mfcc,
                     mfcc_delta,
                     mfcc_delta2,
                     ))
        Y=[CurrentSequence[2]]*Nb_Fenetres
        # on transpose (feature en colonne) et rajoute les lignes correspondant aux nouvelles fenêtres
        Retour_X+=f.transpose().tolist()                
        Retour_Y+=Y
        Retour_NumSequence+=[i]*Nb_Fenetres
    return Retour_NumSequence,np.array(Retour_X),np.array(Retour_Y)

"""
CurrentSequence=Sequences[-1]
Signal=Audio.Signal
hz=Audio.hz
fmin=Audio.fmin
fmax=Audio.fmax
TailleFenetre=1
EcartFenetres=0.5
D.shape[1]
Sequence.shape
import matplotlib.pyplot as plt

for CurrentSequence in Sequences :
    Sequence=Signal[int(CurrentSequence[0].total_seconds()*hz)
                    :int(CurrentSequence[1].total_seconds()*hz)]
    win_l=hz*TailleFenetre
    hop_l=int(win_l/2)
    win_l=int(win_l)
    center=True
    D = np.abs(librosa.stft(Sequence, 
                            window=window, 
                            n_fft=win_l, 
                            win_length=win_l, 
                            hop_length=hop_l,
                            center=center))**2
    print(len(Sequence),win_l,hop_l,D.shape[1])
f.shape
mfcc.shape
len(Retour_X)
len(Retour_Y)
Y.shape
"""