# -*- coding: utf-8 -*-
"""
UE INF a 4-EG : Analyse de données et reconnaissance des formes
@author: Bruno Moreira Nabinger
@author: Clément Vinot

Méthode mixte : Analyse en Composantes Principales (ACP)
              + Classification ascendante hiérarchique (CAH) 
"""

# https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulatio
import numpy as np
import codage
from acp import acp
from cah import cah
import math
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Lecture des données------------------------------------------------------
    # Individus : 28
    noms_individus = readfile("donnees/population_noms_individus.txt")
    # Variables : 10
    noms_variables = readfile("donnees/population_noms_variables.txt")

    mat = np.loadtxt("donnees/population_donnees.txt")
    #print("Matrice", mat.shape,": \n" ,  mat)
    basicStatics(mat)
    matNorm = normalisation(mat)
    print("Analyse en Composantes Principales (ACP)")
    val_p_ind, fact_ind, fact_var = acp(matNorm, noms_individus, 
                                        noms_variables)
    print("I'm back")
    #
    print("Résultat de l'analyse factoriel___________________________________")
    print("Nombre de facteurs = ", len(val_p_ind), "_________________________")
    print("__________________________________________________________________")
    sommeInertiePercentual = 0
    nombrefacteurs = 1,
    flagBreak = 0
    pourcentageInertie = 95
    for nombrefacteurs in range(1,len(val_p_ind)):
        # Inertie = valeur propre / inertie totale 
        print(nombrefacteurs ,"Inertie :", val_p_ind[nombrefacteurs-1],
              "/", matNorm.shape[1], "->",
              np.round(val_p_ind[nombrefacteurs-1] / matNorm.shape[1] * 100,2)
              , "%", "Inertie subtotal", sommeInertiePercentual)
        sommeInertiePercentual = sommeInertiePercentual \
            + np.round(val_p_ind[nombrefacteurs-1] / matNorm.shape[1] * 100,2)
        if(flagBreak == 1):
            break
        if(sommeInertiePercentual > pourcentageInertie):
            print("sommeInertiePercentual > pourcentageInertie")
            print(sommeInertiePercentual, ">", pourcentageInertie)
            flagBreak = 1
            
    print("Nombre de facteurs retenu = ", nombrefacteurs, "__________________")
            
    cah(fact_ind[:,0:nombrefacteurs], noms_individus, noms_variables)
    
    
    cah(fact_var[:,0:nombrefacteurs], noms_variables, noms_individus)