# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:49:50 2018

@author: Pierre
"""
import math
import numpy as np

def Decoupe(Sequences,EcartFenetres,TailleFenetres):
    Retour_Fenetres=[]
    Retour_NumSequence=[]
    Retour_YA=[]
    Retour_YV=[]
    for iSequence in range(len(Sequences)):
        Sequence=Sequences[iSequence]
        LargeurSequence=Sequence[6]
        NbFenetres=1+math.ceil((LargeurSequence-TailleFenetres)/EcartFenetres)
        if NbFenetres==0:
            Ecart=1
        else:
            Ecart=(LargeurSequence-TailleFenetres)/(NbFenetres-1)            
        Retour_Fenetres+=list(np.arange(0,LargeurSequence-TailleFenetres+Ecart/2,Ecart)+Sequence[0].total_seconds())
        Retour_NumSequence+=[iSequence]*NbFenetres
        Retour_YA+=[Sequence[2]]*NbFenetres
        Retour_YV+=[Sequence[3]]*NbFenetres
        if (len(Retour_Fenetres)!=len(Retour_NumSequence)):
            print(iSequence,NbFenetres,
                  np.arange(0,LargeurSequence-TailleFenetres+Ecart,Ecart).shape,
                  LargeurSequence)
    return Retour_NumSequence,np.array(Retour_YA),np.array(Retour_YV),Retour_Fenetres
"""
np.arange(0,7,0.5)

EcartFenetres=0.5
TailleFenetres=1
Sequences=Train
iSequence=31
NbFenetres
len(Retour_Fenetres)
len(Retour_NumSequence)
len(Retour_YA)
len(Retour_YV)
"""