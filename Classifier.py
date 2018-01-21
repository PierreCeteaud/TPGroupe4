# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 23:41:27 2018

@author: Pierre
"""
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

def LDA(Train,Y_Train,Test,Y_Test,Values=(True,False)):
    Clas=LinearDiscriminantAnalysis()
    Clas.fit(Train, Y_Train)    
    PredTrain=Clas.predict(Train)
    PredTest=Clas.predict(Test)
    Retour=[]
    for jeu in ((PredTrain,Y_Train),(PredTest,Y_Test)):
        R=[]
        for Pred in Values:
            for Y in Values:
                R.append(((jeu[0]==Pred)&(jeu[1]==Y)).sum())
        Retour.append(R)
    Retour.append([PredTrain,PredTest])
    Retour.append(Clas)
    return Retour

min_samples_split=0.05

def RF(Train,Y_Train,Test,Y_Test,Values=(True,False)):
    Clas=RandomForestClassifier(min_samples_split=min_samples_split)
    Clas.fit(Train, Y_Train)    
    PredTrain=Clas.predict(Train)
    PredTest=Clas.predict(Test)
    Retour=[]
    for jeu in ((PredTrain,Y_Train),(PredTest,Y_Test)):
        R=[]
        for Pred in Values:
            for Y in Values:
                R.append(((jeu[0]==Pred)&(jeu[1]==Y)).sum())
        Retour.append(R)
    Retour.append([PredTrain,PredTest])
    Retour.append(Clas)
    return Retour

seuilLR=0.5

def LR(Train,Y_Train,Test,Y_Test):
    Values=(True,False)
    Clas=LinearRegression()
    Clas.fit(Train, Y_Train)    
    PredTrain=Clas.predict(Train)>seuilLR
    PredTest=Clas.predict(Test)>seuilLR
    Retour=[]
    for jeu in ((PredTrain,Y_Train),(PredTest,Y_Test)):
        R=[]
        for Pred in Values:
            for Y in Values:
                R.append(((jeu[0]==Pred)&(jeu[1]==Y)).sum())
        Retour.append(R)
    Retour.append([PredTrain,PredTest])
    Retour.append(Clas)
    return Retour

"""
Train=F[1]
Y_Train=Y[1]
Clas.fit(Train, Y_Train)    
print(Train.shape, Y_Train.shape)
Train.min(axis=0)
Y_Train.mean()
"""
# Ne fonctionne qu'avec un LDA à deux valeurs cibles
def Print(Result):
    for R in(("Train -",Result[0]),("Test -",Result[1])):
        if R[1][0]==0:
            print(R[0],"Préc :",f"{0}%",
              "Rappel :",f"{0}%",
              "Erreur :",f"{round((R[1][1]+R[1][2])/sum(R[1]),1)}%")
        else:
            print(R[0],f"Préc :",f"{round(R[1][0]/(R[1][0]+R[1][1])*100,1)}%",
              "Rappel :",f"{round(R[1][0]/(R[1][0]+R[1][2])*100,1)}%",
              "Erreur :",f"{round((R[1][1]+R[1][2])/sum(R[1])*100,1)}%")

    #print(1-((PredTrain==Audio_Y).sum()/Audio_Y.size),1-((PredTest==Audio_TY).sum()/Audio_TY.size))
    
def PredictionFenetres(NumSequence,Reel,Predictions):    
    Ok=0
    Ko=0
    Classe=-1
    SequenceActuelle=-1
    PredictionsFenetres=[]
    for NSeq,R,Pred in zip(NumSequence,Reel,Predictions):
        if Classe==-1:
            Classe=R
            Votes=np.zeros(4)
            SequenceActuelle=NSeq
            Votes[Pred]=1
        else:
            if SequenceActuelle==NSeq:
                Votes[Pred]+=1
            else:
                maxi=0
                Choix=-1
                for i in range(4):
                    if Votes[i]>maxi:
                        Choix=i;
                        maxi=Votes[i]
                PredictionsFenetres.append(Choix)
                if Choix==Classe:
                    Ok+=1
                else:
#                    print(SequenceActuelle,Votes,Choix)
                    Ko+=1
                Classe=R
                Votes=np.zeros(4)
                Votes[Pred]=1
                SequenceActuelle=NSeq    
    maxi=0
    Choix=-1
    for i in range(4):
        if Votes[i]>maxi:
            Choix=i;
            maxi=Votes[i]
    PredictionsFenetres.append(Choix)
    if Choix==Classe:
        Ok+=1
    else:
        Ko+=1
    return PredictionsFenetres,Ok,Ko    
    
"""
NumSequence=NumSeqAudioTrain
Reel=Classe
Predictions=Pred_Test

Tmp=9
NSeq=NumSequence[9]
R=Reel[9]
Pred=Predictions[1]
Tmp+=1
"""

def PredictionFenetre4Classifieurs(NumSequence,Reel,Predictions):
    Ok=0
    Ko=0
    Classe=-1
    SequenceActuelle=-1
    PredictionsFenetres=[]
    for NSeq,R,Pred in zip(NumSequence,Reel,Predictions):
        if Classe==-1:
            Classe=R
            Votes=np.zeros(4)
            VotesEgalité=np.zeros(4)
            SequenceActuelle=NSeq
            if sum(Pred)!=1:
                for i in range(4):
                    if Pred[i]==1:
                        VotesEgalité[i]=1
            else:                
                Votes[Pred.index(1)]=1
        else:
            if SequenceActuelle==NSeq:
                if sum(Pred)!=1:
                    for i in range(4):
                        if Pred[i]==1:
                            VotesEgalité[i]+=1
                else:
                    Votes[Pred.index(1)]+=1
            else:
                maxi=0
                Choix=-1
                Egalités=[]
                for i in range(4):
                    if Votes[i]>maxi:
                        Choix=i;
                        maxi=Votes[i]
                        Egalités=[i]
                    elif Votes[i]==maxi:
                        Egalités.append(i)
                if (len(Egalités)!=1):
                    if len(VotesEgalité)>1:
                        maxi=0
                        for Candidat in Egalités:
                            if VotesEgalité[Candidat]>maxi:
                                Choix=Candidat;
                                maxi=VotesEgalité[Candidat]
                PredictionsFenetres.append(Choix)
                if Choix==Classe:
                    Ok+=1
                else:
                    Ko+=1
                Classe=R
                Votes=np.zeros(4)
                VotesEgalité=np.zeros(4)
                if sum(Pred)!=1:
                    for i in range(4):
                        if Pred[i]==1:
                            VotesEgalité[i]=1
                else:                
                    Votes[Pred.index(1)]=1
                SequenceActuelle=NSeq    
    maxi=0
    Choix=-1
    Egalités=[]
    for i in range(4):
        if Votes[i]>maxi:
            Choix=i;
            maxi=Votes[i]
            Egalités=[i]
        elif Votes[i]==maxi:
            Egalités.append(i)
    if (len(Egalités)!=1):
        if len(VotesEgalité)>1:
            maxi=0
            for Candidat in Egalités:
                if VotesEgalité[Candidat]>maxi:
                    Choix=Candidat;
                    maxi=VotesEgalité[Candidat]
    PredictionsFenetres.append(Choix)
    if Choix==Classe:
        Ok+=1
    else:
        Ko+=1
    return PredictionsFenetres,Ok,Ko

from statsmodels import robust

Mediane_=0
Mad_=0

def NormaliseTrain(Features):
    global Mediane_
    global Mad_
    Mediane_=np.median(Features,axis=0)
    #Mediane_=np.mean(Features,axis=0)
    Mad_=robust.mad(Features,axis=0)
    #Mad_=np.std(Features,axis=0)
    Mad_[np.isnan(Mad_)]=1
    Mad_[Mad_==0]=1
    return NormaliseAutres(Features)
def NormaliseAutres(Features):
    return (Features-Mediane_)/Mad_
"""    

Normalisation(Features)
NumSequence=NumSeqAudioTrain
Reel=Classe

Tmp=0
NSeq=NumSequence[Tmp]
R=Reel[Tmp]
Pred=Predictions[Tmp]
Tmp+=1
"""