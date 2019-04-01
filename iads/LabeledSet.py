# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: LabeledSet.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""
# ---------------------------
import numpy as np
import pandas as pd
import math

# ---------------------------
class LabeledSet:
    """ Classe pour représenter un ensemble d'exemples (base d'apprentissage)
        Variables d'instance :
            - input_dimension (int) : dimension de la description d'un exemple (x)
            - nb_examples (int) : nombre d'exemples dans l'ensemble
    """

    def __init__(self, input_dimension):
        """ Constructeur de LabeledSet
            Argument:
                - intput_dimension (int) : dimension de x
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.nb_examples = 0

    def addExample(self,vector,label):
        """ Ajout d'un exemple dans l'ensemble
            Argument:
                - vector ()
                - label (int) : classe de l'exemple (+1 ou -1)

        """
        if (self.nb_examples == 0):
            self.x = np.array([vector])
            self.y = np.array([label])
        else:
            self.x = np.vstack((self.x, vector))
            self.y = np.vstack((self.y, label))

        self.nb_examples = self.nb_examples + 1

    def getInputDimension(self):
        """ Renvoie la dimension de l'espace d'entrée
        """
        return self.input_dimension

    def size(self):
        """ Renvoie le nombre d'exemples dans l'ensemble
        """
        return self.nb_examples

    def getX(self, i):
        """ Renvoie la description du i-eme exemple (x_i)
        """
        return self.x[i]

    #
    def getY(self, i):
        """ Renvoie la classe de du i-eme exemple (y_i)
        """
        return(self.y[i])

    def split(self, i):
        indice = math.ceil(self.size()*i)
        ls1 = LabeledSet(self.input_dimension)
        ls2 = LabeledSet(self.input_dimension)
        for i in range(indice):
            ls1.addExample(self.getX(i), self.getY(i))
        for i in range(indice, self.size()):
            ls2.addExample(self.getX(i), self.getY(i))
        return ls1, ls2
