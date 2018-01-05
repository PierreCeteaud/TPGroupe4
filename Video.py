# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:47:18 2017

@author: Pierre
"""

import cv2
import numpy as np
import math
fps=25

def Train_Video(Sequences,EcartFenetres,TailleFenetre,hz,center=True,cadree=False):
    # EcartFenetres et TailleFenetre sont donnés en secondes
    # Retour_Xne une liste des features par fenetre
    # une ligne par fenêtre
    # une colonne par feature, la dernière colonne indique si la fenêtre est avec ou sans la présentatrice    
    # le nombre de fenêtres suit la règle définie par librosa.sftf
    Retour_X=[]
    Retour_Y=[]
    for CurrentSequence in Sequences :
        # Temps en s de la sequence
        LargeurSequence=CurrentSequence[6]
        if LargeurSequence>=TailleFenetre:
            # Nombre de fenêtres dans cette séquence 
            if center:
                # La formule a été trouvée empiriquement
                NbFenetres=math.ceil((LargeurSequence+1/hz)/EcartFenetres)
            else:
                # Celle-ci aussi
                NbFenetres=math.ceil((LargeurSequence-TailleFenetre+1/hz)/EcartFenetres)
            # Comme le nombre de fenêtre ne tombe pas pile poil pour les espacer de EcartFenetres
            # à l'intérieur de la séquence on recalcule EcartFenetre et on l'appelle EspacementFenetre
            # On n'a pas la garantie que sftf garde les fenêtre à l'intérieur de la séquence
            if NbFenetres==1:
                # Cas très spécifique où la séquence fait exactement la taille d'une fenêtre
                EspacementFenetres=0
            else:
                EspacementFenetres=(LargeurSequence-TailleFenetre)/(NbFenetres-1)
            for iFenetre in range (NbFenetres):
                ImageMilieu=int(fps*(CurrentSequence[0].total_seconds()+iFenetre*EspacementFenetres))
                img=GetImage(ImageMilieu,cadree)
                # Les deux première features sont
                # la moyenne et la variance sur le niveau de gris
                grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                m = np.mean(grayImage)
                v= np.var(grayImage)
                R=[m,v]
                # On rajoute 6 features :
                # La moyenne et la variance sur chaque couleur
                for color in range(3):
                    image=img[:,:,color]                    
                    m = np.mean(image)
                    v= np.var(image)
                    R.append(m)
                    R.append(v)
                # on va faire à peut prêt la même chose 
                # avec le dodage des couleurs hsv
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # pour la saturation et la luminosité c'est exactement pareil
                for val in range(1,3):
                    image=img[:,:,color]                    
                    m = np.mean(image)
                    v= np.var(image)
                    R.append(m)
                    R.append(v)
                # par contre la teinte est un angle
                # d'une façon générale pour un angle la moyenne de 2 et 358 est 0 et non 180
                # et de plus openCV code les angles entre 0 et 179
                # On doit donc trouver moyenne de 178 et 2 =0
                angles=hsv[:,:,0]/180*math.pi
                x=np.cos(angles)
                y=np.sin(angles)
                xmoyen=x.mean()
                ymoyen=y.mean()
                #Le calcul de l'angle moyen doit être adapté en fonction du 
                #quadrant  où  se  trouve  le  vecteur  moyen,
                if xmoyen==0:
                    if ymoyen>=0:
                        m=math.pi/2
                    else:
                        m=-math.pi/2
                elif xmoyen>0:
                    m=math.atan(ymoyen/xmoyen)
                else: 
                    m=math.pi+math.atan(ymoyen/xmoyen)
                R.append(m)
                v=1-(np.var(x)+np.var(y))**0.5
                R.append(v)
                Retour_X.append(R)        
                Retour_Y.append(CurrentSequence[3])
    return np.asarray(Retour_X),np.asarray(Retour_Y)


def GetImage(NumImage,cadree=False):
    if cadree:
        Name=f"Images/cadree-{NumImage}.png"        
    else:
        Name=f"Images/06-11-22-{NumImage}.png"
    return cv2.imread(Name)
"""
CurrentSequence=Sequences[0]
EcartFenetres=0.5
TailleFenetre=1
center=True
hz=22050
iFenetre=0
cadree=True
"""