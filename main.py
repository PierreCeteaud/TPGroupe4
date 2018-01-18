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

## Lecture du fichier des séquences
dateReference=datetime(1900,1,1)
Sequences=[] # Seulement les séquences assez longues
ToutesSequences=[] # Peu être utile si on souhaite regouper des séquences pour n'étudier que l'audio par exemple
with open("Présente.csv",'r') as FSequences:
    for line in FSequences:       
        texte=line.split(";")        
        try:
            Candidat=[datetime.strptime(texte[0],"%M:%S,%f")-dateReference,
                  datetime.strptime(texte[1],"%M:%S,%f")-dateReference,
                  int(texte[2]), # Présentatrice parle
                  int(texte[3]), # Présentatrice à l'écran
                  int(texte[4]), # Changement de séquence dû à l'audio
                  int(texte[5]), # Changement de séquence dû à la vidéo                  
                  ]
            Ok=True
        except: 
        # Si erreur au décodage (1ère et dernière ligne) on ignore
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

importlib.reload(Classifier)

Normalisation=True
if Normalisation:
    Features=Classifier.NormaliseTrain(Features)
    TestFeatures=Classifier.NormaliseAutres(TestFeatures)

# Concaténation des classes audio et vidéo
Both_Y=Audio_Y*2+Video_Y
Both_TY=Audio_TY*2+Video_TY

# Classification globale
G=Classifier.LDA(Features, Both_Y,TestFeatures,Both_TY,(0,1,2,3))
print("Un classifieur")
print("Taux d'erreur sur le train:",1-(G[0][0]+G[0][5]+G[0][10]+G[0][15])/sum(G[0]))
print("Taux d'erreur sur le test:",1-(G[1][0]+G[1][5]+G[1][10]+G[1][15])/sum(G[1]))

# C'est moins bien :-) => on garde les entrées videos pour améliorer l
Y11=(Audio_Y==1)&(Video_Y==1)
Y10=(Audio_Y==1)&(Video_Y==0)
Y01=(Audio_Y==0)&(Video_Y==1)
Y00=(Audio_Y==0)&(Video_Y==0)
TY11=(Audio_TY==1)&(Video_TY==1)
TY10=(Audio_TY==1)&(Video_TY==0)
TY01=(Audio_TY==0)&(Video_TY==1)
TY00=(Audio_TY==0)&(Video_TY==0)

Liste_Resultats=[]
for F in (("Audio+Video",Features,TestFeatures),("Audio",Audio_Features,Audio_Test_Features),("Video",Video_Features,Video_Test_Features)):
    for Y in (("Audio",Audio_Y,Audio_TY),
              ("Video",Video_Y,Video_TY),
              ("Totalement absente",Y00,TY00),
              ("Présente video",Y01,TY01),
              ("Présente audio",Y10,TY01),
              ("Présente Audio+Video",Y11,TY11)):           
        print(Y[0],"grâce à",F[0])
        Result=Classifier.LDA(F[1],Y[1],F[2],Y[2])
        Liste_Resultats.append(Result)
        Classifier.Print(Result)

# Nombre d'erreurs avec deux classifeurs
print("Deux classifieurs")        
print("Taux d'erreur sur le train:",1-((Audio_Y==Liste_Resultats[0][2][0])
                                    &(Video_Y==Liste_Resultats[1][2][0])).sum()/len(Audio_Y))

print("Taux d'erreur sur le test:",1-((Audio_TY==Liste_Resultats[0][2][1])
                                    &(Video_TY==Liste_Resultats[1][2][1])).sum()/len(Audio_TY))

# Nombre d'erreurs avec quatre classifeurs
print("Quatre classifieurs")
TrainPredit00=Liste_Resultats[2][2][0].astype(int)
Test_Predit00=Liste_Resultats[2][2][1].astype(int)
TrainPredit0V=Liste_Resultats[3][2][0].astype(int)
Test_Predit0V=Liste_Resultats[3][2][1].astype(int)
TrainPreditA0=Liste_Resultats[4][2][0].astype(int)
Test_PreditA0=Liste_Resultats[4][2][1].astype(int)
TrainPreditAV=Liste_Resultats[5][2][0].astype(int)
Test_PreditAV=Liste_Resultats[5][2][1].astype(int)

TrainPredit=(TrainPredit00+TrainPredit0V+TrainPreditA0+TrainPreditAV)==1
TrainOk=(TrainPredit&((TrainPredit00&Y00)|(TrainPredit0V&Y01)|(TrainPreditA0&Y10)|(TrainPreditAV&Y11))).sum()
Test_Predit=(Test_Predit00+Test_Predit0V+Test_PreditA0+Test_PreditAV)==1
Test_Ok=(Test_Predit&((Test_Predit00&TY00)|(Test_Predit0V&TY01)|(Test_PreditA0&TY10)|(Test_PreditAV&TY11))).sum()

print("Train : % avec une prédiction",TrainPredit.sum()/len(TrainPredit))
print("Train : % erreur sur prédiction",1-TrainOk/TrainPredit.sum())
print("Train : % erreur sur l'ensemble",1-TrainOk/len(TrainPredit))
print("Test  : % avec une prédiction",Test_Predit.sum()/len(Test_Predit))
print("Test  : % erreur sur prédiction",1-Test_Ok/Test_Predit.sum())
print("Test  : % erreur sur l'ensemble",1-Test_Ok/len(Test_Predit))

print("Fenetres :")

print("Un classifieur")
Pred_Train=G[2][0]
Pred_Test=G[2][1]

Classe=Audio_Y*2+Video_Y # 0= absente A et V, 1 présente vidéo, 2 présente Audio, 3 présente audio et vidéo
PredictionsFenetres,Ok,Ko=Classifier.PredictionFenetres(NumSeqTrain,Classe,Pred_Train)
print("Taux d'erreur sur le train :",Ko/(Ok+Ko))

Classe=Audio_TY*2+Video_TY # 0= absente A et V, 1 présente vidéo, 2 présente Audio, 3 présente audio et vidéo
PredictionsFenetres,Ok,Ko=Classifier.PredictionFenetres(NumSeqTest,Classe,Pred_Test)
print("Taux d'erreur sur le test :",Ko/(Ok+Ko))

print("Deux classifieurs")
Pred_Train=Liste_Resultats[0][2][0]*2+Liste_Resultats[1][2][0]
Pred_Test=Liste_Resultats[0][2][1]*2+Liste_Resultats[1][2][1]

Classe=Audio_Y*2+Video_Y # 0= absente A et V, 1 présente vidéo, 2 présente Audio, 3 présente audio et vidéo
PredictionsFenetres,Ok,Ko=Classifier.PredictionFenetres(NumSeqTrain,Classe,Pred_Train)
print("Taux d'erreur sur le train :",Ko/(Ok+Ko))

#importlib.reload(Classifier)
Classe=Audio_TY*2+Video_TY # 0= absente A et V, 1 présente vidéo, 2 présente Audio, 3 présente audio et vidéo
PredictionsFenetres,Ok,Ko=Classifier.PredictionFenetres(NumSeqTest,Classe,Pred_Test)
print("Taux d'erreur sur le test :",Ko/(Ok+Ko))

print("Quatre classifieurs")

Classe=Audio_Y*2+Video_Y # 0= absente A et V, 1 présente vidéo, 2 présente Audio, 3 présente audio et vidéo
Predictions=list(zip(TrainPredit00,TrainPredit0V,TrainPreditA0,TrainPreditAV))

PredictionsFenetres,Ok,Ko=Classifier.PredictionFenetre4Classifieurs(NumSeqTrain,Classe,Predictions)

print("Taux d'erreur sur le train :",Ko/(Ok+Ko))
Classe=Audio_TY*2+Video_TY # 0= absente A et V, 1 présente vidéo, 2 présente Audio, 3 présente audio et vidéo
Predictions=list(zip(Test_Predit00,Test_Predit0V,Test_PreditA0,Test_PreditAV))

PredictionsFenetres,Ok,Ko=Classifier.PredictionFenetre4Classifieurs(NumSeqTest,Classe,Predictions)

print("Taux d'erreur sur le test :",Ko/(Ok+Ko))

"""


len(Train)
len(Video_Features[1])
print(len(Audio_Features),len(Audio_Y))
print(len(Video_Features),len(Video_Y))
import importlib
importlib.reload(Audio)
importlib.reload(Video)
importlib.reload(Classifier)
len(Audio_Features[0])
# Test 4
List_Erreurs=[1,2,7,12,24,25,36,39,49,50,56,61,64,69,72,73,77,80,92,93]
# Test 1
List_Erreurs=[1,3,11,22,28,32,33,36,40,41,44,45,48,49,57,58,62,72,75,83,88,91,101,103,110]
List_Erreurs=[1,3]
for E in List_Erreurs:
    D=NumSeqAudioTest.index(E)
    F=NumSeqAudioTest.index(E+1)
    test=slice(D,F)
    print(Test[E][0].total_seconds(),Test[E][1].total_seconds())
    print(Pred_Test[test])
    print(PredictionsFenetres[E])
    print(Classe[test])

Classe[0:20] 
NumSeqAudioTest[0:20]
NumSeqAudioTest[test]
Audio_TY[test]
Video_TY[test]
Test_Predit00[test]
Test_Predit0V[test]
Test_PreditA0[test]
Test_PreditAV[test]
"""