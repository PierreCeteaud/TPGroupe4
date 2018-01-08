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
n_mfcc=13
try:
    print("On conserve le signal à",hz,"hz")
except:
    Signal,hz= librosa.load('06-11-22.wav')


def Features_Audio(Fenetres,TailleFenetre,EcartSousFenetres,NbSignals=10,center=True):
    # EcartFenetres et TailleFenetre sont donnés en secondes
    # une ligne par fenêtre
    # une colonne par feature
    
    # Retour_X une liste des features par fenetre
    Retour_X=[]
    win_l=hz*TailleFenetre/NbSignals
    hop_l=int(win_l*EcartSousFenetres)
    win_l=int(win_l)
    for DebutFenetre in Fenetres:
        Sequence=Signal[int(DebutFenetre*hz):int(DebutFenetre*hz+TailleFenetre*hz)]
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
        # Intégration temporelle
        mfcc = np.mean(mfcc,axis=1)
        mfcc_delta = np.mean(mfcc_delta,axis=1)
        mfcc_delta2 = np.mean(mfcc_delta2,axis=1)
        
        # Concatenation des features         
        f=np.hstack((mfcc,
                     mfcc_delta,
                     mfcc_delta2,
                     ))
        # on transpose (feature en colonne) et rajoute les lignes correspondant aux nouvelles fenêtres
        Retour_X.append(f.tolist())             
    return np.array(Retour_X)

"""
Signal=Audio.Signal
hz=Audio.hz
fmin=Audio.fmin
fmax=Audio.fmax
Fenetres=FenetresTrain
NbSignals=10
center=True
TailleFenetre=1
EcartSousFenetres=0.5

DebutFenetre=Fenetres[0]

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