# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:41:00 2017

@author: Pierre
"""

def Train_Audio(Sequences,EcartFenetres,TailleFenetre):
    # EcartFenetres et TailleFenetre sont donnés en secondes
    # Retourne une liste des features par fenetre
    # une ligne par fenêtre
    # une colonne par feature, la dernière colonne indique si la fenêtre est avec ou sans la présentatrice
    # Sur chaque séquence l'écart est ajusté légèrement à la baisse pour faire entrer un nombre entier de fenêtres
    return [[]]