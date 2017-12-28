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
import matplotlib.pyplot as plt

window = 'hamming'
fmin = 20
# Cette limite est à étudier, en effet on trouvera facilement que la voix humaine
# varie entre 80 et 1500 hz
fmax = 4000
Signal,hz= librosa.load('06-11-22.wav')


def Train_Audio(Sequences,EcartFenetres,TailleFenetre):
    # EcartFenetres et TailleFenetre sont donnés en secondes
    # Retourne une liste des features par fenetre
    # une ligne par fenêtre
    # une colonne par feature, la dernière colonne indique si la fenêtre est avec ou sans la présentatrice
    # Sur chaque séquence l'écart est ajusté légèrement à la baisse pour faire entrer un nombre entier de fenêtres
    Retour=[]
    for CurrentSequence in Sequences :
        Sequence=Signal[int(CurrentSequence[0].total_seconds()*hz)
                        :int(CurrentSequence[1].total_seconds()*hz)]
        win_l=hz*TailleFenetre
        hop_l=int(win_l/2)
        win_l=int(win_l)
        center=False
        D = np.abs(librosa.stft(Sequence, 
                                window=window, 
                                n_fft=win_l, 
                                win_length=win_l, 
                                hop_length=hop_l,
                                center=center))**2
        
        # calcul du MEL
        S = feature.melspectrogram(S=D, y=Sequence, n_mels=24, fmin=fmin, fmax=fmax)
        # calcul des 13 coefficients
        feats = feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)
        # Rajout de la ligne de présence
        Nb_Fenetres=feats.shape[1]
        f=np.vstack([feats,np.asarray([CurrentSequence[2]]*Nb_Fenetres).reshape(1,Nb_Fenetres)])
        # on transpose (feature en colonne) et rojoute les lignes correspondant aux nouvelles fenêtres
        Retour+=f.transpose().tolist()                
    return Retour

