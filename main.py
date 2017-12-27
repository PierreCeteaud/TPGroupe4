# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:06:55 2017

@author: Pierre
"""

from datetime import timedelta
from datetime import datetime
from Audio import Train_Audio
from Video import Train_Video

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
                # Filtre sur les séquences trop courtes 
                if Candidat[6]>=1:
                    Sequences.append(Candidat)
                
# On choisi de faire l'apprentissage sur les séquences longues et les tests sur les séquences courtes pour :
#   Avoir beaucoup de fenêtres d'apprentissage (une fenêtre = une ligne pour le classifieur)
#   Avoir beaucoup de jeux de test (la validité du modèle est bien sur l'identification d'une séquence)

ratio=0.5 # On fait un 50/50 en nombre de séquences
limit=int(len(Sequences)*ratio)
Sequences.sort(key=lambda colonnes: colonnes[6])
Test=Sequences[0:limit]
Train=Sequences[limit+1:]

Audio_Features=Train_Audio(Train,0.5,1)
Video_Features=Train_Video(Train,0.5,1)

# test
Audio_Features=[[1,2,0],[3,4,1],[4,5,0],[6,7,1]]
Video_Features=[[8,0],[0,0],[2,1],[4,1]]

# Concaténation des features audio et cideo, les deux dernières colonnes 
# correspondent à l'identification de la présentatrice (audio/video)
Features=[A[:-1]+V[:-1]+[A[-1]]+[V[-1]] for A,V in zip(Audio_Features,Video_Features)]
