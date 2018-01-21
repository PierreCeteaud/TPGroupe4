# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:06:55 2017

@author: Pierre
"""

from datetime import timedelta
from datetime import datetime
import Audio
import Video 
import Classifier
import Fenetrage
import numpy as np
import importlib
import sys

## Lecture du fichier des séquences
dateReference=datetime(1900,1,1)
Sequences=[] # Seulement les séquences assez longues
ToutesSequences=[] # Peu être utile si on souhaite regouper des séquences pour n'étudier que l'audio par exemple
with open("Présente.csv",'r') as FSequences:
    for line in FSequences:       
        texte=line.split(";")        
        try:
            Candidat=[datetime.strptime(texte[0],"%H:%M:%S,%f")-dateReference,
                  datetime.strptime(texte[1],"%H:%M:%S,%f")-dateReference,
                  int(texte[2]), # Présentatrice parle
                  int(texte[3]), # Présentatrice à l'écran
                  int(texte[4]), # Changement de séquence dû à l'audio
                  int(texte[5]), # Changement de séquence dû à la vidéo                  
                  ]
            Ok=True
        except : 
        # Si erreur au décodage (1ère et dernière ligne) on ignore
            print(sys.exc_info())
            Ok=False
        if Ok :
            # On rajoute une colonne avec la durée de la séquence
            Candidat.append((Candidat[1]-Candidat[0]).total_seconds())
            # On suppose que l'on connait le début de la présentation et la fin du débat
            if Candidat[0]>timedelta(0,84) and Candidat[0]<timedelta(0,2340.5):
                ToutesSequences.append(Candidat)
                # Filtre sur les séquences trop courtes et sur la présence de plusieurs personnes
                if Candidat[6]>=1 and Candidat[2]<2 and Candidat[3]<2:
                    Sequences.append(Candidat)
                

from sklearn.model_selection import train_test_split

ratio=0.7
Train, Test = train_test_split(Sequences, train_size=ratio, random_state = 42)  

importlib.reload(Fenetrage)
NumSeqTrain,Audio_Y,Video_Y,FenetresTrain=Fenetrage.Decoupe(Train,0.5,1)
NumSeqTest,Audio_TY,Video_TY,FenetresTest=Fenetrage.Decoupe(Test,0.5,1)

importlib.reload(Audio)
Audio_Features=Audio.Features_Audio(FenetresTrain,1,0.5,center=False)
importlib.reload(Video)
Video_Features=Video.Features_Video(FenetresTrain,1,cadree=True)

Audio_Test_Features=Audio.Features_Audio(FenetresTest,1,0.5,center=False)
Video_Test_Features=Video.Features_Video(FenetresTest,1,cadree=True)

# Concaténation et normalisation des features audio et Video


Features=np.hstack((Audio_Features,Video_Features))
TestFeatures=np.hstack((Audio_Test_Features,Video_Test_Features))

Features=Classifier.NormaliseTrain(Features)
TestFeatures=Classifier.NormaliseAutres(TestFeatures)

# On équilibre les échantillons à 50/50 d'audio

Features00=Features[(Audio_Y==0) & (Video_Y==0)]
Features01=Features[(Audio_Y==0) & (Video_Y==1)]
Features10=Features[(Audio_Y==1) & (Video_Y==0)]
Features11=Features[(Audio_Y==1) & (Video_Y==1)]
Limit=min(Features00.shape[0],Features01.shape[0],Features10.shape[0],Features11.shape[0])
Features00=Features00[0:Limit]
Features01=Features01[0:Limit]
Features10=Features10[0:Limit]
Features11=Features11[0:Limit]
AY00=Audio_Y[(Audio_Y==0) & (Video_Y==0)][0:Limit]
AY01=Audio_Y[(Audio_Y==0) & (Video_Y==1)][0:Limit]
AY10=Audio_Y[(Audio_Y==1) & (Video_Y==0)][0:Limit]
AY11=Audio_Y[(Audio_Y==1) & (Video_Y==1)][0:Limit]
FeaturesEA=np.vstack((Features00,Features01,Features10,Features11))
AY=np.hstack((AY00,AY01,AY10,AY11))
TestFeatures00=TestFeatures[(Audio_TY==0) & (Video_TY==0)]
TestFeatures01=TestFeatures[(Audio_TY==0) & (Video_TY==1)]
TestFeatures10=TestFeatures[(Audio_TY==1) & (Video_TY==0)]
TestFeatures11=TestFeatures[(Audio_TY==1) & (Video_TY==1)]
Limit=min(TestFeatures00.shape[0],TestFeatures01.shape[0],TestFeatures10.shape[0],TestFeatures11.shape[0])
TestFeatures00=TestFeatures00[0:Limit]
TestFeatures01=TestFeatures01[0:Limit]
TestFeatures10=TestFeatures10[0:Limit]
TestFeatures11=TestFeatures11[0:Limit]
ATY00=Audio_TY[(Audio_TY==0) & (Video_TY==0)][0:Limit]
ATY01=Audio_TY[(Audio_TY==0) & (Video_TY==1)][0:Limit]
ATY10=Audio_TY[(Audio_TY==1) & (Video_TY==0)][0:Limit]
ATY11=Audio_TY[(Audio_TY==1) & (Video_TY==1)][0:Limit]
TestFeaturesEA=np.vstack((TestFeatures00,TestFeatures01,TestFeatures10,TestFeatures11))
ATY=np.hstack((ATY00,ATY01,ATY10,ATY11))


"""
Features[:,0:41].shape
Audio_Features.shape
np.array_equal(Audio_Features,Features[:,0:41])
NAF=(Audio_Features-Classifier.Mediane_[0:41])/Classifier.Mad_[0:41]
importlib.reload(Classifier)
np.array_equal(NAF,Features[:,0:41])
"""

# Classification globale

Liste_Resultats=[]
F =("Audio+Video",FeaturesEA,TestFeaturesEA)
Y =("Audio",AY,ATY)
F[1].shape
Y[1].shape
Result=Classifier.LR(F[1],Y[1],F[2],Y[2])
Liste_Resultats.append(Result)
Classifier.Print(Result)
"""
np.argwhere(np.isnan(Features))
np.argwhere(np.isnan(Video_Features))
np.argwhere(np.isnan(Audio_Features))
Classifier.Mad_[84]
Classifier.Mediane_[84]
Classifier.NormaliseTrain(Features)[:,84:88]
"""
# Nombre d'erreurs avec deux classifeurs
print("Deux classifieurs")        
print("Taux d'erreur sur le train:",1-((Audio_Y==Liste_Resultats[0][2][0])
                                    &(Video_Y==Liste_Resultats[1][2][0])).sum()/len(Audio_Y))

print("Taux d'erreur sur le test:",1-((Audio_TY==Liste_Resultats[0][2][1])
                                    &(Video_TY==Liste_Resultats[1][2][1])).sum()/len(Audio_TY))

Labels=[]
suf=""
for d in range(3):
    for i in range(Audio.n_mfcc):
        Labels.append("mfcc"+suf+str(i))
    suf=suf+"'"
Labels.append("ZCR")
Labels.append("Spectral contrats")
for i in ("gris","bleu","vert","rouge","saturation","luminosité","teinte"):
    for j in ("moyenne", "variance"):
        Labels.append(j+" "+i)
Labels.append("Dynamisme")
for i in ("bleu","teinte","vert","saturation","rouge","luminosité"):
    for j in range (16):
        Labels.append("Histogramme "+i+" "+str(j))
        
ImportanceAudio=list(zip(Labels,(Liste_Resultats[0][3].feature_importances_).reshape(len(Labels))))
ImportanceAudio=list(zip(Labels,(abs(Liste_Resultats[0][3].coef_)).reshape(len(Labels))))
ImportanceVideo=list(zip(Labels,(abs(Liste_Resultats[1][3].scalings_*Liste_Resultats[1][3].coef_.T)).reshape(len(Labels))))
ScalingsAudio=list(zip(Labels,(abs(Liste_Resultats[0][3].scalings_)).reshape(len(Labels))))
ImportanceAudio=list(zip(Labels,(Liste_Resultats[0][3].explained_variance_ratio_).reshape(len(Labels))))
ImportanceVideo=list(zip(Labels,(abs(Liste_Resultats[1][3].scalings_*Liste_Resultats[1][3].coef_.T)).reshape(len(Labels))))
abs(Liste_Resultats[0][3].scalings_*Liste_Resultats[0][3].coef_[0,0])

len(Labels)
Audio_Features.shape
Video_Features.shape
Liste_Resultats[0][3].coef_.shape
np.median(Features,axis=0)
np.mean(Features,axis=0)
from statsmodels import robust
robust.mad(Features,axis=0)

