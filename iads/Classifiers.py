# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
import random

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système
        """
        cptBon=0
        for i in range(0, dataset.size()):
            if (self.predict(dataset.getX(i))*dataset.getY(i) > 0):
                cptBon+=1
        return cptBon/dataset.size()

    def loss(self,labeledSet):
        somme = 0
        for i in labeledSet.size():
            x = labeledSet.getX(i)
            y = labeledSet.getY(i)
            somme += (y - self.predict(x))**2
        return somme

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.w= -1 + np.random.sample(input_dimension)*2

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        return np.vdot(x,self.w)

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        print("Pas d'apprentissage pour ce classifieur")

# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.k = k
        self.dim=input_dimension

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        def dist(u,v):
            dist = (u-v)**2
            dist = sum(dist)
            return dist

        d = np.array([dist(x,self.labeledSet.getX(i)) for i in range(self.labeledSet.size())])
        dSort=np.argsort(d)
        pos = 0
        neg = 0
        for i in range(self.k):
            if (self.labeledSet.getY(dSort[i])<0):
                neg+=1
            else:
                pos+=1

        return (pos-neg)/self.k


    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.labeledSet = labeledSet

# ---------------------------

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dim = input_dimension
        self.learning_rate = learning_rate
        self.w = np.array([0]*self.dim)

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        return z


    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        tab = ([k for k in range(labeledSet.size())])
        random.shuffle(tab)
        for i in tab:
            x = labeledSet.getX(i)
            y = labeledSet.getY(i)
            self.w=self.w+self.learning_rate*(y-np.sign(self.predict(x)))*x


# ---------------------------

class ClassifierGradientStochastique(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dim = input_dimension
        self.learning_rate = learning_rate
        self.w = np.array([0]*self.dim)

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        return z

    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        tab = ([k for k in range(labeledSet.size())])
        random.shuffle(tab)
        for i in tab:
            x = labeledSet.getX(i)
            y = labeledSet.getY(i)
            self.w = self.w+self.learning_rate*(y-self.predict(x))*x

# ---------------------------

class ClassifierGradientBatch(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dim = input_dimension
        self.learning_rate = learning_rate
        self.w = np.array([0]*self.dim)

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        return z


    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        tab = ([k for k in range(labeledSet.size())])
        random.shuffle(tab)
        gradient = np.array([0.] * self.dim)
        for i in tab:
            x = labeledSet.getX(i)
            y = labeledSet.getY(i)
            gradient += (y - self.predict(x)) * x
        self.w = self.w + self.learning_rate * gradient

# ---------------------------

class ClassifierPerceptronKernel(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dim = dimension_kernel
        self.learning_rate = learning_rate
        self.kernel=kernel
        self.w = np.array([0]*self.dim)

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        xtrans = self.kernel.transform(x)
        z = np.dot(xtrans, self.w)
        return z


    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        tab = ([k for k in range(labeledSet.size())])
        random.shuffle(tab)
        for i in tab:
            x = labeledSet.getX(i)
            xtrans = self.kernel.transform(x)
            y = labeledSet.getY(i)
            self.w=self.w+self.learning_rate*(y-np.sign(self.predict(x)))*xtrans
