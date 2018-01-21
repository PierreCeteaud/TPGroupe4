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
Audio_Y_tmp=np.asarray(Audio_Y)
Audio_Y0=Audio_Y_tmp[Audio_Y_tmp==0]
Audio_Y1=Audio_Y_tmp[Audio_Y_tmp==1]


importlib.reload(Audio)
Audio_Features=Audio.Features_Audio(FenetresTrain,1,0.5,center=False)
importlib.reload(Video)
Video_Features=Video.Features_Video(FenetresTrain,1,cadree=True)

Audio_Test_Features=Audio.Features_Audio(FenetresTest,1,0.5,center=False)
Video_Test_Features=Video.Features_Video(FenetresTest,1,cadree=True)

# Concaténation et normalisation des features audio et Video
Features=np.hstack((Audio_Features,Video_Features))
TestFeatures=np.hstack((Audio_Test_Features,Video_Test_Features))

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

LabelsA=[]
suf=""
for d in range(3):
    for i in range(Audio.n_mfcc):
        LabelsA.append("mfcc"+suf+str(i))
    suf=suf+"'"
LabelsA.append("ZCR")
LabelsA.append("Spectral contrats")
LabelsV=[]
for i in ("gris","bleu","vert","rouge","saturation","luminosité","teinte"):
    for j in ("moyenne", "variance"):
        LabelsV.append(j+" "+i)
Labels.append("Dynamisme")
for i in ("bleu","teinte","vert","saturation","rouge","luminosité"):
    for j in range (16):
        LabelsV.append("Histogramme "+i+" "+str(j))
LabelsAV=LabelsA+LabelsV

Importance1=list(zip(LabelsAV,abs(G[3].coef_.T)))
fichier = open("Importance1.txt", "w")
fichier.write("Feature\tN\tA\tV\tAV\n")
for ligne in Importance1:
    fichier.write(f"{ligne[0]}\t{ligne[1][0]}\t{ligne[1][1]}\t{ligne[1][2]}\t{ligne[1][3]}\n")

fichier.close()
    
importlib.reload(Classifier)
Liste_Resultats=[]
for F in (("Audio+Video",Features,TestFeatures),("Audio",Audio_Features,Audio_Test_Features),("Video",Video_Features,Video_Test_Features)):
    for Y in (("Audio",Audio_Y,Audio_TY),
              ("Video",Video_Y,Video_TY)):           
              #("Totalement absente",Y00,TY00),
              #("Présente video",Y01,TY01),
              #("Présente audio",Y10,TY01),
              #("Présente Audio+Video",Y11,TY11)
        #print(Y[0],"grâce à",F[0])
        Result=Classifier.LR(F[1],Y[1],F[2],Y[2])
        Liste_Resultats.append(Result)
        if F=="Audio+Video":
            Importance=list(zip(LabelsAV,abs(Result[3].coef_.T)))
        elif F=="Audio":
            Importance=list(zip(LabelsA,abs(Result[3].coef_.T)))
        else:
            Importance=list(zip(LabelsV,abs(Result[3].coef_.T)))
        fichier = open(f"Importance_{F[0]}={Y[0]}.txt", "w")
        fichier.write("Feature\tImportance\n")
        for ligne in Importance:
            fichier.write(f"{ligne[0]}\t{ligne[1]}\n")
        fichier.close()

            #Classifier.Print(Result)




# Nombre d'erreurs avec deux classifeurs
print("Deux classifieurs")        
print("Taux d'erreur sur le train:",1-((Audio_Y==Liste_Resultats[0][2][0])
                                    &(Video_Y==Liste_Resultats[1][2][0])).sum()/len(Audio_Y))

print("Taux d'erreur sur le test:",1-((Audio_TY==Liste_Resultats[0][2][1])
                                    &(Video_TY==Liste_Resultats[1][2][1])).sum()/len(Audio_TY))


# Matrices de confusion
from sklearn.metrics import confusion_matrix
print ("1ère approche")
CM=confusion_matrix(Both_TY,G[2][1])
Classifier.plot_confusion_matrix(CM)
plt.show()
Classifier.plot_confusion_matrix(CM,True)
plt.show()

print ("2ème approche")
Aggregation=Liste_Resultats[0][2][1]*2+Liste_Resultats[1][2][1]
CM=confusion_matrix(Both_TY,Aggregation)
Classifier.plot_confusion_matrix(CM)
plt.show()
Classifier.plot_confusion_matrix(CM,True)
plt.show()

print ("3ème approche")
Aggregation=Liste_Resultats[2][2][1]*2+Liste_Resultats[5][2][1]
CM=confusion_matrix(Both_TY,Aggregation)
Classifier.plot_confusion_matrix(CM)
plt.show()
Classifier.plot_confusion_matrix(CM,True)
plt.show()

print ("4ème approche")
A_A=Liste_Resultats[2]
A_V=Liste_Resultats[3]
V_A=Liste_Resultats[4]
V_V=Liste_Resultats[5]
PoidsInterne=2
Seuil=(PoidsInterne+1)/2
RAudio=(A_A[3].predict(Audio_Test_Features)*PoidsInterne+V_A[3].predict(Video_Test_Features))>Seuil
RVideo=(V_V[3].predict(Video_Test_Features)*PoidsInterne+A_V[3].predict(Audio_Test_Features))>Seuil
Aggregation=RAudio*2+RVideo
CM=confusion_matrix(Both_TY,Aggregation)
Classifier.plot_confusion_matrix(CM)
plt.show()
Classifier.plot_confusion_matrix(CM,True)
plt.show()

        
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