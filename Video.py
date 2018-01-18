# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:47:18 2017

@author: Pierre
"""

import cv2
import numpy as np
import math
fps=25

Images=[None]*58773
 


def Features_Video(Fenetres,TailleFenetre,cadree=True):
    # EcartFenetres et TailleFenetre sont donnés en secondes
    # Retour_Xne une liste des features par fenetre
    # une ligne par fenêtre
    # une colonne par feature, la dernière colonne indique si la fenêtre est avec ou sans la présentatrice    
    # le nombre de fenêtres suit la règle définie par librosa.sftf
    Retour_X=[]
    for DebutFenetre in Fenetres:
        ImageMilieu=int(fps*(DebutFenetre+TailleFenetre/2))
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
        D=Dynamisme(DebutFenetre,TailleFenetre)
        R.append(D)
        Retour_X.append(R)   
        if len(Retour_X)!=Fenetres.index(DebutFenetre)+1:
            print(len(Retour_X),Fenetres.index(DebutFenetre))
    return np.asarray(Retour_X)

def Dynamisme(DebutFenetre,TailleFenetre):
    ImagePrec=None
    Mouvements=None
    for Iimage in range(int(DebutFenetre*fps),int((DebutFenetre+TailleFenetre)*fps)):
        Image=cv2.cvtColor(GetImage(Iimage)[0:200,5:-5], cv2.COLOR_BGR2GRAY)
        #Image=GetImage(Iimage)[0:200,5:-5]
        #plt.imshow(cv2.cvtColor(Image,cv2.COLOR_BGR2RGB))
        #plt.show()
        if ImagePrec is None:
#            print(Image.shape)
            Mouvements=np.zeros(Image.shape,dtype=int)
        else:
            Mouvements+=np.bitwise_xor(Image,ImagePrec)
        ImagePrec=Image
    Mvt=Mouvements
    Fond=np.hstack((Mvt[0:100,0:75],Mvt[0:100,235:310]))
    FondFlat=Fond.flatten()
    FondFlat.sort()
    SeuilBruit=FondFlat[14970]
    SeuilBruit=min(np.mean(Fond)+np.std(Fond)*2,np.max(Fond))
    Mv=np.copy(Mvt)
    Mv=Mv-SeuilBruit
    Mv[Mv<0]=0
    Mv[0:100,0:75]=0
    Mv[0:100,235:310]=0
    return Mv.sum() # On pourrait diviser par le nombre d'image, le nombre de pixels or fondet par 3 
                    # pour avoir une moyenne cohérente avec l'écart 0-255
"""
    Mvt=np.sum(Mouvements,axis=2)
    
    np.max(Mv)/11/3
    plt.imshow(Mvt,cmap='gray')
    plt.imshow(Mv,cmap='gray')
    c=Mvt-Mv
    plt.imshow(c,cmap='gray')
    type (Mvt[0,0])
"""
def GetImage(NumImage,cadree=False):
    if Images[NumImage] is None:
        Name=f"Images/06-11-22-{NumImage}.png"        
        Images[NumImage]=cv2.imread(Name)
        if Images[NumImage] is None:
            print(f"Image :{NumImage} absente" )
            while Images[NumImage] is None:
                NumImage-=1
    if cadree:        
        return Images[NumImage][15:200,40:-40]    
    return Images[NumImage]
"""
def GetImage(NumImage,cadree=False):
    if cadree:
        Name=f"Images/cadree-{NumImage}.png"        
    else:
        Name=f"Images/06-11-22-{NumImage}.png"
    return cv2.imread(Name)
DebutFenetre=16*60+36
TailleFenetre=1
CurrentSequence=Sequences[100]
EcartFenetres=0.25
TailleFenetre=1
center=False
hz=22050
iFenetre=0
cadree=True
NumImage=45544
"""