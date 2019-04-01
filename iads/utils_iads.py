# -*- coding: utf-8 -*-

"""
Package: siads
Fichier: utils.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de 3i026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importation de LabeledSet
from . import LabeledSet as ls

def plot2DSet(set):
    """ LabeledSet -> NoneType
        Hypothèse: set est de dimension 2
        affiche une représentation graphique du LabeledSet
        remarque: l'ordre des labels dans set peut être quelconque
    """
    S_pos = set.x[np.where(set.y == 1),:][0]      # tous les exemples de label +1
    S_neg = set.x[np.where(set.y == -1),:][0]     # tous les exemples de label -1
    plt.scatter(S_pos[:,0],S_pos[:,1],marker='o') # 'o' pour la classe +1
    plt.scatter(S_neg[:,0],S_neg[:,1],marker='x') # 'x' pour la classe -1

def plot_frontiere(set,classifier,step=10):
    """ LabeledSet * Classifier * int -> NoneType
        Remarque: le 3e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=set.x.max(0)
    mmin=set.x.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])
    
# ------------------------ 

def createGaussianDataset(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ 
        rend un LabeledSet 2D généré aléatoirement.
        Arguments:
        - positive_center (vecteur taille 2): centre de la gaussienne des points positifs
        - positive_sigma (matrice 2*2): variance de la gaussienne des points positifs
        - negative_center (vecteur taille 2): centre de la gaussienne des points négative
        - negative_sigma (matrice 2*2): variance de la gaussienne des points négative
        - nb_points (int):  nombre de points de chaque classe à générer
    """
    labeledSet = ls.LabeledSet(2)
    for i in range(0, nb_points):
        pos = np.random.multivariate_normal(positive_center,positive_sigma)
        neg = np.random.multivariate_normal(negative_center,negative_sigma)
        labeledSet.addExample(pos,1)
        labeledSet.addExample(neg,-1)
    return labeledSet
    

def createXOR(n, var):
    labeledSet = ls.LabeledSet(2)
    for i in range(0, n):
        labeledSet.addExample(np.random.multivariate_normal([0,0],[[var,0],[0,var]]),1)
        labeledSet.addExample(np.random.multivariate_normal([1,1],[[var,0],[0,var]]),1)
        labeledSet.addExample(np.random.multivariate_normal([1,0],[[var,0],[0,var]]),-1)
        labeledSet.addExample(np.random.multivariate_normal([0,1],[[var,0],[0,var]]),-1)
    return labeledSet

# ------------------------ 

class KernelBias:
    def transform(self,x):
        y=np.asarray([x[0],x[1],1])
        return y

class KernelPoly:
    def transform(self,x):
        y=np.asarray([1, x[0], x[1], x[0]*x[0], x[1]*x[1], x[0]*x[1]])
        return y

