# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 11:20:22 2017

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
try:
    print("On conserve le signal à",hz,"hz")
except:
    Signal,hz= librosa.load('06-11-22.wav')


def Train_Audio(Sequences,EcartFenetres,TailleFenetre,center=True):
    # EcartFenetres et TailleFenetre sont donnés en secondes
    # Retourne une liste des features par fenetre
    # une ligne par fenêtre
    # une colonne par feature, la dernière colonne indique si la fenêtre est avec ou sans la présentatrice
    # le nombre de fenêtres suit la règle définie par librosa.sftf
    Retour=[]
    win_l=hz*TailleFenetre
    hop_l=int(win_l/2)
    win_l=int(win_l)
    for CurrentSequence,Num in zip(Sequences,range(len(Sequences))) :
        Sequence=Signal[int(CurrentSequence[0].total_seconds()*hz)
                        :int((CurrentSequence[0].total_seconds()+CurrentSequence[1])*hz)]
        D = np.abs(librosa.stft(Sequence, 
                                window=window, 
                                n_fft=win_l, 
                                win_length=win_l, 
                                hop_length=hop_l,
                                center=center))**2
        
        # calcul du MEL
        S = feature.melspectrogram(S=D, y=Sequence, n_mels=24, fmin=fmin, fmax=fmax)
        # calcul des 13 coefficients
        mfcc = feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)
        # Calcul de la dérivée
        mfcc_delta = librosa.feature.delta(mfcc)
        # Calcul de la dérivée seconde
        mfcc_delta2 = librosa.feature.delta(mfcc_delta)

        # Concatenation des features et Rajout de la ligne de présence
        Nb_Fenetres=mfcc.shape[1]
        f=np.vstack((mfcc,
                     mfcc_delta,
                     mfcc_delta2,
                     np.asarray([Num]*Nb_Fenetres).reshape(1,Nb_Fenetres),
                     np.asarray([CurrentSequence[3]]*Nb_Fenetres).reshape(1,Nb_Fenetres),
                     ))
        # on transpose (feature en colonne) et rajoute les lignes correspondant aux nouvelles fenêtres
        Retour+=f.transpose().tolist()                
    return Retour

from datetime import timedelta
from datetime import datetime

dateReference=datetime(1900,1,1)
Sequences=[] # Seulement les séquences assez longues
ToutesSequences=[] # Peu être utile si on souhaite regouper des séquences pour n'étudier que l'audio par exemple
with open("SequencesAudio.csv",'r') as FSequences:
    for line in FSequences:       
        texte=line.split(";")
        try:
            Duree=datetime.strptime(texte[1],"%M:%S,%f")
            Duree=Duree.minute*60+Duree.second+Duree.microsecond /1000000
            Candidat=[datetime.strptime(texte[0],"%M:%S,%f")-dateReference,
                  Duree,
                  texte[2], # Nom
                  int(texte[3]), # Type
                  ]
            Ok=True
        except: 
        # Si erreur au décodage (1ère et dernière ligne) on ignore 
            #print(sys.exc_info())
            Ok=False        
        if Ok :
            # On suppose que l'on connait le début de la présentation et la fin du débat
            if Candidat[0]>timedelta(0,84) and Candidat[0]<timedelta(0,2340.5):
                ToutesSequences.append(Candidat)
                # Filtre sur les séquences trop courtes et sur la présence de plusieurs personnes
                if  Candidat[1]>=1 and Candidat[3]>0:
                    Sequences.append(Candidat)

ratio=0.5 # On fait un 50/50 en nombre de séquences
limit=int(len(Sequences)*ratio)
Sequences.sort(key=lambda colonnes: colonnes[1])
Test=Sequences[0:limit]
Train=Sequences[limit+1:]


fAudio_Train=Train_Audio(Train,0.5,1,center=True)
fAudio_Test=Train_Audio(Test,0.5,1,center=True)


len(fAudio_Train)
len(Sequences)
len(ToutesSequences)

    
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA=LinearDiscriminantAnalysis()
XTrain=np.array(fAudio_Train)[:,:-2]
YTrain=np.array(fAudio_Train)[:,-1]
LDA.fit(XTrain, YTrain)
PredTrain=LDA.predict(XTrain)
print("Erreur de prédiction sur train :",1-((PredTrain==YTrain).sum()/YTrain.size))
XTest=np.array(fAudio_Test)[:,:-2]
YTest=np.array(fAudio_Test)[:,-1]
PredTest=LDA.predict(XTest)
print("Erreur de prédiction sur test :",1-((PredTest==YTest).sum()/YTest.size))

Ok=0
Ko=0
POk=0
PKo=0
Interlocuteur=-1
NumSequence=-1
for Sequence,Pred in zip(fAudio_Test,PredTest):
    Pred=int(Pred)
    if Interlocuteur==-1:
        Interlocuteur=Sequence[-1]
        Votes=np.zeros(6)
        Votes[Pred]=1
        NumSequence=Sequence[-2]
        if Sequence[-1]!=Pred:
            print("#",Interlocuteur,Pred)
            PKo+=1
        else:
            POk+=1
    else:
        if NumSequence==Sequence[-2]:
            Votes[Pred]+=1
        else:
            maxi=0
            Choix=-1
            print(Votes)
            for i in range(1,6):
                if Votes[i]>maxi:
                    Choix=i;
                    maxi=Votes[i]
                elif Votes[i]==maxi:
                    Choix=-1                
            print(Choix,Interlocuteur)
            if Choix==Interlocuteur:
                Ok+=1
            else:
                Ko+=1
            Interlocuteur=Sequence[-1]
            Votes=np.zeros(6)
            Votes[Pred]+=1
            NumSequence=Sequence[-2]
        if Sequence[-1]!=Pred:
            print("#",Interlocuteur,Pred)
            PKo+=1
        else:
            POk+=1

maxi=0
Choix=-1
print(Votes)
for i in range(1,6):
    if Votes[i]>maxi:
        Choix=i;
        maxi=Votes[i]
    elif Votes[i]==maxi:
        Choix=-1                
print(Choix,Interlocuteur)
if Choix==Interlocuteur:
    Ok+=1
else:
    Ko+=1

            
print(Ok,Ko,Ko/(Ok+Ko))

vars(LDA)
LDA.coef_.shape
Importance=abs(LDA.coef_)
MaxCoef=Importance.max()
Importance=Importance*255/MaxCoef

Importance=LDA.coef_
MaxCoef=Importance.max()
MinCoef=Importance.min()
Importance=(Importance-MinCoef)*255/(MaxCoef-MinCoef)


np.array(fAudio_Train).shape
YTrain.shape

import matplotlib.pyplot as plt

plt.imshow(Importance,cmap='gray')

len(fAudio_Test)