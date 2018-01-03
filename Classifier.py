# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 23:41:27 2018

@author: Pierre
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA(Train,Y_Train,Test,Y_Test,Values=(True,False)):
    LDA=LinearDiscriminantAnalysis()
    LDA.fit(Train, Y_Train)    
    PredTrain=LDA.predict(Train)
    PredTest=LDA.predict(Test)
    Retour=[]
    for jeu in ((PredTrain,Y_Train),(PredTest,Y_Test)):
        R=[]
        for Pred in Values:
            for Y in Values:
                R.append(((jeu[0]==Pred)&(jeu[1]==Y)).sum())
        Retour.append(R)
    Retour.append([PredTrain,PredTest])
    return Retour

# Ne fonctionne qu'avec un LDA à deux valeurs cibles
def Print(Result):
    for R in(("Train -",Result[0]),("Test -",Result[1])):
        if R[1][0]==0:
            print(R[0],"Préc :",0,
              "Rappel :",0,
              "Erreur :",round((R[1][1]+R[1][2])/sum(R[1]),1))
        else:
            print(R[0],f"Préc :",round(R[1][0]/(R[1][0]+R[1][1])*100,1),
              "Rappel :",round(R[1][0]/(R[1][0]+R[1][2])*100,1),
              "Erreur :",round((R[1][1]+R[1][2])/sum(R[1])*100,1))

    #print(1-((PredTrain==Audio_Y).sum()/Audio_Y.size),1-((PredTest==Audio_TY).sum()/Audio_TY.size))
    

"""    
Train=Features
Y_Train= Audio_Y
Test=TestFeatures
Y_Test=Audio_TY

jeu=(PredTrain,Y_Train)

Pred=True
Y=True
Pred=False
Y=False
jeu=(PredTest,Y_Test)

R=("Train :",Result[0])
"""