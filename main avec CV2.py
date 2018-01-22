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
import random
import math
from sklearn.metrics import f1_score


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

ratio=2/3
Train, T1 = train_test_split(Train, train_size=ratio, random_state = 44)  

ratio=0.5
T3, T2 = train_test_split(Train, train_size=ratio, random_state = 44)  


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
LabelsV.append("Dynamisme")
for i in ("bleu","teinte","vert","saturation","rouge","luminosité"):
    for j in range (16):
        LabelsV.append("Histogramme "+i+" "+str(j))
LabelsAV=LabelsA+LabelsV

importlib.reload(Video)
#Video_Features=Video.Features_Video(FenetresTrain,1,cadree=True)
#Video_Test_Features=Video.Features_Video(FenetresTest,1,cadree=True)

importlib.reload(Audio)
TailleFenetre=4/3
Audio.fmin=100/3
Audio.fmax=933
Audio.n_mfcc=24
Audio.n_MEL=36
f_anal=31
Recouvrement=2
Best_F1=0
Last_TailleFenetre=0
Ampleur=1
IndexTest=0
NbPossibilités=14
Dispo=np.ones((NbPossibilités))
Mieux=False
OldTailleFenetre=TailleFenetre
Oldfmin=Audio.fmin
Oldfmax=Audio.fmax
Oldn_mfcc=Audio.n_mfcc
Oldn_MEL=Audio.n_MEL
Oldf_anal=f_anal
OldRecouvrement=Recouvrement

while True:    
        if TailleFenetre!=Last_TailleFenetre:
            importlib.reload(Fenetrage)
            NumSeqT1,Audio_Y1,Video_Y1,F1=Fenetrage.Decoupe(T1,0.5,TailleFenetre)
            NumSeqT2,Audio_Y2,Video_Y2,F2=Fenetrage.Decoupe(T2,0.5,TailleFenetre)
            NumSeqT3,Audio_Y3,Video_Y3,F3=Fenetrage.Decoupe(T3,0.5,TailleFenetre)
            Last_TailleFenetre=TailleFenetre

        Audio_F1=Audio.Features_Audio(F1,TailleFenetre,
                                            1/Recouvrement,center=False,
                                            fen_anal=f_anal)
        Audio_F2=Audio.Features_Audio(F2,TailleFenetre,
                                             1/Recouvrement,
                                             center=False,fen_anal=f_anal)
        Audio_F3=Audio.Features_Audio(F3,TailleFenetre,
                                             1/Recouvrement,
                                             center=False,fen_anal=f_anal)


        # Concaténation et normalisation des features audio et Video
        #Features=np.hstack((Audio_Features,Video_Features))
        #TestFeatures=np.hstack((Audio_Test_Features,Video_Test_Features))
            
        Fea12=np.copy(np.vstack((Audio_F1,Audio_F2)))
        Fea23=np.copy(np.vstack((Audio_F2,Audio_F3)))
        Fea13=np.copy(np.vstack((Audio_F1,Audio_F3)))
        Y12=np.hstack((Audio_Y1,Audio_Y2))
        Y23=np.hstack((Audio_Y2,Audio_Y3))
        Y13=np.hstack((Audio_Y1,Audio_Y3))
        Fea12=Classifier.NormaliseTrain(Fea12)
        Audio_F3=Classifier.NormaliseAutres(Audio_F3)
        Fea23=Classifier.NormaliseTrain(Fea23)
        Audio_F1=Classifier.NormaliseAutres(Audio_F1)
        Fea13=Classifier.NormaliseTrain(Fea13)
        Audio_F2=Classifier.NormaliseAutres(Audio_F2)

        Liste_Resultats=[]
        F1S=0
        for F,Y in zip([("1-2",Fea12,Audio_F3),
                        ("2-3",Fea23,Audio_F1),
                        ("1-3",Fea13,Audio_F2)],
                       [("3",Y12,Audio_Y3),
                        ("1",Y23,Audio_Y1),
                        ("2",Y13,Audio_Y2)]):
            Result=Classifier.LR(F[1],Y[1],F[2],Y[2])
            F1S+=f1_score(Y[2], Result[2][1])
        F1S=F1S/3
        if F1S>Best_F1:
            print(TailleFenetre,Audio.fmin,Audio.fmax,Audio.n_mfcc,Audio.n_MEL,f_anal,Recouvrement)
            print("Audio-Audio :",F1S)
            Best_F1=F1S
            Mieux=True
            OldTailleFenetre=TailleFenetre
            Oldfmin=Audio.fmin
            Oldfmax=Audio.fmax
            Oldn_mfcc=Audio.n_mfcc
            Oldn_MEL=Audio.n_MEL
            Oldf_anal=f_anal
            OldRecouvrement=Recouvrement
        else:
            TailleFenetre=OldTailleFenetre
            Audio.fmin=Oldfmin
            Audio.fmax=Oldfmax
            Audio.n_mfcc=Oldn_mfcc
            Audio.n_MEL=Oldn_MEL
            f_anal=Oldf_anal
            Recouvrement=OldRecouvrement
                        
        Choix=random.randint(1,NbPossibilités)
        while Dispo[Choix-1]==0:
            Choix=random.randint(1,NbPossibilités)
        print(Choix,end=' ')
        if Choix==10:
            if TailleFenetre==2:
                TailleFenetre/=1.2**Ampleur
            TailleFenetre*=1.5**Ampleur
            if TailleFenetre>2:
                TailleFenetre=2
        elif Choix==1:
            TailleFenetre/=1.5**Ampleur
        elif Choix==2:
            Audio.fmin*=2**Ampleur
        elif Choix==3:
            if 1000/f_anal==Audio.fmin:
                Audio.f_anal*=1.5**Ampleur
            else:
                Audio.fmin/=2**Ampleur
                if 1000/f_anal>Audio.fmin:
                    Audio.fmin=1000/f_anal+1
        elif Choix==4:
            Audio.fmax*=2**Ampleur
        elif Choix==5:
            Audio.fmax/=2**Ampleur
        elif Choix==6:
            if Audio.n_mfcc==Audio.n_MEL:
                Audio.n_MEL+=1
            else:
                Audio.n_mfcc+=max(1,int(math.ceil(Audio.n_mfcc*(1.5**Ampleur-1))))
                if Audio.n_mfcc>Audio.n_MEL:
                    Audio.n_mfcc=Audio.n_MEL
        elif Choix==7:
            Audio.n_mfcc-=max(1,int(math.ceil(Audio.n_mfcc*(1.5**Ampleur-1))))
            if Audio.n_mfcc<1:
                Audio.n_mfcc=1
        elif Choix==8:
            Audio.n_MEL+=max(1,int(math.ceil(Audio.n_MEL*(1.5**Ampleur-1))))
        elif Choix==9:
            if Audio.n_mfcc==Audio.n_MEL:
                if Audio.n_mfcc>1:
                    Audio.n_mfcc-=1
            else:
                Audio.n_MEL-=max(1,int(math.ceil(Audio.n_MEL*(1.5**Ampleur-1))))
                if Audio.n_MEL<Audio.n_mfcc:
                    Audio.n_MEL=Audio.n_mfcc
                elif Audio.n_MEL<1:
                    Audio.n_MEL=1
        elif Choix==11:
            f_anal*=2**Ampleur
        elif Choix==12:
            if 1000/f_anal==Audio.fmin:
                Audio.fmin/=1.5**Ampleur
            else:
                f_anal/=2**Ampleur
                if 1000/f_anal>Audio.fmin:
                    f_anal=1000/Audio.fmin+1
        elif Choix==13:
            Recouvrement+=Ampleur
        elif Choix==14:
            Recouvrement-=Ampleur
            if Recouvrement<=1:
                Recouvrement=1                
        Dispo[Choix-1]=0
        if sum(Dispo)==0:
            Dispo=np.ones((NbPossibilités))
            if not Mieux:
                Ampleur=Ampleur/2
                print(Ampleur)
            else:
                Ampleur=min(Ampleur*1.1,1)
                print('-',end=' ')
            Mieux=False

NumSeqTrain,Audio_Y,Video_Y,FenetresTrain=Fenetrage.Decoupe(Train,0.5,TailleFenetre)
NumSeqTest,Audio_TY,Video_TY,FenetresTest=Fenetrage.Decoupe(Test,0.5,TailleFenetre)

Audio_Test_Features=Audio.Features_Audio(FenetresTest,TailleFenetre,
                                         1/Recouvrement,center=False,
                                         fen_anal=f_anal)
Audio_Features=Audio.Features_Audio(FenetresTrain,TailleFenetre,
                                    1/Recouvrement,center=False,
                                    fen_anal=f_anal)

Features=Classifier.NormaliseTrain(Audio_Features)
TestFeatures=Classifier.NormaliseAutres(Audio_Test_Features)

F=("Audio",Features,TestFeatures)
Y=("Audio",Audio_Y,Audio_TY)

Result=Classifier.LR(F[1],Y[1],F[2],Y[2])
f1_score(Y[2], Result[2][1])
 
    
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