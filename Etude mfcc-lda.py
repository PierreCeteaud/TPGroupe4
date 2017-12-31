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
fmax = 4000
n_mfcc=13
# Cette limite est à étudier, en effet on trouvera facilement que la voix humaine
# varie entre 80 et 1500 hz
fmin = 97
fmax = 4361

fmin = 158
fmax = 4750

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
                     np.asarray([Num]*Nb_Fenetres).reshape(1,Nb_Fenetres),
                     np.asarray([CurrentSequence[3]]*Nb_Fenetres).reshape(1,Nb_Fenetres),
                     ))
        # on transpose (feature en colonne) et rajoute les lignes correspondant aux nouvelles fenêtres
        Retour+=f.transpose().tolist()                
    return Retour

def ReussiteFenetre(fAudio,Prediction):
    Ok=0
    Ko=0
    POk=0
    PKo=0
    Interlocuteur=-1
    NumSequence=-1
    for Sequence,Pred in zip(fAudio,Prediction):
        Pred=int(Pred)
        if Interlocuteur==-1:
            Interlocuteur=Sequence[-1]
            Votes=np.zeros(6)
            Votes[Pred]=1
            NumSequence=Sequence[-2]
            if Sequence[-1]!=Pred:
                PKo+=1
            else:
                POk+=1
        else:
            if NumSequence==Sequence[-2]:
                Votes[Pred]+=1
            else:
                maxi=0
                Choix=-1
                for i in range(1,6):
                    if Votes[i]>maxi:
                        Choix=i;
                        maxi=Votes[i]
                    elif Votes[i]==maxi:
                        Choix=-1                
                if Choix==Interlocuteur:
                    Ok+=1
                else:
                    Ko+=1
                Interlocuteur=Sequence[-1]
                Votes=np.zeros(6)
                Votes[Pred]+=1
                NumSequence=Sequence[-2]
            if Sequence[-1]!=Pred:
                PKo+=1
            else:
                POk+=1
    
    maxi=0
    Choix=-1
    for i in range(1,6):
        if Votes[i]>maxi:
            Choix=i;
            maxi=Votes[i]
        elif Votes[i]==maxi:
            Choix=-1                
    if Choix==Interlocuteur:
        Ok+=1
    else:
        Ko+=1
    return Ok,Ko


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
"""
ratio=0.5 # On fait un 50/50 en nombre de séquences
limit=int(len(Sequences)*ratio)
Sequences.sort(key=lambda colonnes: colonnes[1])
Test=Sequences[0:limit]
Train=Sequences[limit+1:]
"""
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ratio=0.7
Train, Test = train_test_split(Sequences, train_size=ratio, random_state = 42)  

Result=[]
Coefs=[]
for n_mfcc in range(25,26):
    fAudio_Train=Train_Audio(Train,0.5,1,center=True)
    fAudio_Test=Train_Audio(Test,0.5,1,center=True)
    LDA=LinearDiscriminantAnalysis()
    XTrain=np.array(fAudio_Train)[:,:-2]
    YTrain=np.array(fAudio_Train)[:,-1]
    LDA.fit(XTrain, YTrain)    
    PredTrain=LDA.predict(XTrain)
    R=[n_mfcc]
    R.append(1-((PredTrain==YTrain).sum()/YTrain.size))
    XTest=np.array(fAudio_Test)[:,:-2]
    YTest=np.array(fAudio_Test)[:,-1]
    PredTest=LDA.predict(XTest)
    R.append(1-((PredTest==YTest).sum()/YTest.size))
    Ok,Ko=ReussiteFenetre(fAudio_Train,PredTrain)
    R.append(Ko/(Ok+Ko))
    Ok,Ko=ReussiteFenetre(fAudio_Test,PredTest)
    R.append(Ko/(Ok+Ko))
    Result.append(R)
    Importance=abs(LDA.coef_)
    Importance=Importance.sum(axis=0)/Importance.sum()
    Coefs.append(Importance)
    
    
    
LDA.coef_.shape
Importance=abs(LDA.coef_)
print(Importance.sum(axis=0)/Importance.sum())
MaxCoef=Importance.max()
Importance=Importance*255/MaxCoef


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

MaxCoef=XTrain.max()
MinCoef=XTrain.min()
XTrain.shape
pltXTrain=(XTrain-MinCoef)*255/(MaxCoef-MinCoef)
pltXTrain.shape

dpi = 80
margin = 0.05 # (5% of the width/height of the figure...)
xpixels, ypixels = 800, 800

# Make a figure big enough to accomodate an axis of xpixels by ypixels
# as well as the ticklabels, etc...
figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi

fig = plt.figure(figsize=figsize, dpi=dpi)
# Make the axis the right size...
ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
ax.imshow(pltXTrain,cmap='gray',extent=[0,75,0,len(XTrain)],aspect ='auto')
#ax.imshow(Importance,cmap='gray',extent=[0,75,0,len(Importance)],aspect ='auto')
plt.show()
len(fAudio_Test)
Importance.shape

Importance=abs(LDA.coef_)
Importance=Importance.sum(axis=0)/Importance.sum()
print((Importance).reshape(len(Importance),1))

file = open("Result_mfcc.csv", "w")
for R in Result:
    for value in R:
        file.write(str(value))
        file.write(";")
    file.write("\n")
file.close()

file = open("mfcc_Coef.csv", "w")
for C in Coefs:
    print(len(C))
    file.write(str(len(C)))
    file.write(";")
    for value in C:
        file.write(str(value))
        file.write(";")
    file.write("\n")
file.close()

len(Coefs)
len(Result)
len(Coefs[1])