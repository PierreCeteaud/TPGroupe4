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
import numpy as np

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

Audio_Features,Audio_Y=Audio.Train_Audio(Train,0.5,1)
Video_Features,Video_Y=Video.Train_Video(Train,0.5,1,Audio.hz,cadree=True)
if len(Video_Features)!=len(Audio_Features):
    print("Erreur sur la synchronisation des fenêtres audios/video")


# Concaténation des features audio et Video

Features=np.hstack((Audio_Features,Video_Features))


Audio_Test_Features,Audio_TY=Audio.Train_Audio(Test,0.5,1)
Video_Test_Features,Video_TY=Video.Train_Video(Test,0.5,1,Audio.hz,cadree=True)
TestFeatures=np.hstack((Audio_Test_Features,Video_Test_Features))

# Concaténation des classement audio et vidéo
Both_Y=Audio_Y*2+Video_Y
Both_TY=Audio_TY*2+Video_TY



# Classification globale

importlib.reload(Classifier)
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
        #Classifier.Print(Result)
        
# Nombre d'erreurs avec deux classifeurs
print("Deux classifieurs")        
print("Taux d'erreur sur le train:",1-((Audio_Y==Liste_Resultats[0][2][0])
                                    &(Video_Y==Liste_Resultats[1][2][0])).sum()/len(Audio_Y))

print("Taux d'erreur sur le test:",1-((Audio_TY==Liste_Resultats[0][2][1])
                                    &(Video_TY==Liste_Resultats[1][2][1])).sum()/len(Audio_TY))

# Nombre d'erreurs avec quatre classifeurs
print("Quatre classifieurs")
TrainPredit00=Liste_Resultats[2][2][0]
Test_Predit00=Liste_Resultats[2][2][1]
TrainPredit0V=Liste_Resultats[3][2][0]
Test_Predit0V=Liste_Resultats[3][2][1]
TrainPreditA0=Liste_Resultats[4][2][0]
Test_PreditA0=Liste_Resultats[4][2][1]
TrainPreditAV=Liste_Resultats[5][2][0]
Test_PreditAV=Liste_Resultats[5][2][1]

TrainPredit=Liste_Resultats[2][2][0],axis=0)

print("Taux d'erreur sur le train:",1-((Audio_Y==Liste_Resultats[0][2][0])
                                    &(Video_Y==Liste_Resultats[1][2][0])).sum()/len(Audio_Y))

print("Taux d'erreur sur le test:",1-((Audio_TY==Liste_Resultats[0][2][1])
                                    &(Video_TY==Liste_Resultats[1][2][1])).sum()/len(Audio_TY))



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
"""