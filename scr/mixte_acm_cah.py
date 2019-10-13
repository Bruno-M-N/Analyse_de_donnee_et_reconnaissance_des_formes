# -*- coding: utf-8 -*-
"""
UE INF a 4-EG : Analyse de données et reconnaissance des formes
@author: Bruno Moreira Nabinger
@author: Clément Vinot

Méthode mixte : Analyse factorielle des correspondances multiples (ACM)
              + Classification ascendante hiérarchique (CAH) 
"""

# https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulation
import numpy as np
import codage
from acm import acm
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
    
    # Affiche les noms des variables 
    print("            ", end='')
    for i in range(len(noms_variables)):
        print(noms_variables[i].rstrip('\n'), "  ", end='')
    print('')
    # Affiche les noms des individus et les valeurs associés 
    for i in range(mat.shape[0]):
        ligne = mat[i,:]
        # rstrip('\n') remove trailing '\n' 
        print(noms_individus[i].rstrip('\n'),"     ", np.array2string(ligne,
              formatter={'float_kind': lambda ligne: "%6.1f" % ligne}))
    
    print("_"*82)
    #  Étude univariée qui consiste à décrire individuellement chaque variable
    basicStatics(mat)
    
    print("_"*82)
    nClasses = 4
    print("Mise en classes : Nombre de Classes =", nClasses)
    mat = quantitatif_en_qualitatif1(mat, nClasses)
    print("_"*82)
    
    # Affiche les noms des variables
    print("            ", end='')
    for i in range(len(noms_variables)):
        print(noms_variables[i].rstrip('\n'), "  ", end='')
    print('')
    # Affiche les noms des individus et les classes associés
    for i in range(mat.shape[0]):
        ligne = mat[i,:]
        print(noms_individus[i].rstrip('\n'),"     ", np.array2string(ligne,
              formatter={'int': lambda ligne: "%6.0f" % ligne}))
        
    print("_"*82)
    print("Analyse factorielle des correspondances multiples (ACM)")
    noms_modalites, val_p_mod1, fact_mod1, fact_mod2 = acm(mat, noms_individus,
                                                           noms_variables)
    
    print("Résultat de l'analyse factoriel")
    print("Nombre de facteurs = ", len(val_p_mod1))

    sommeInertiePercentual = 0
    nombrefacteurs = len(val_p_mod1)
    flagBreak = 0
    pourcentageInertie = 0.95
#    for nombrefacteurs in range(1,len(val_p_mod1)):
#        # Inertie = valeur propre / inertie totale 
##        print(nombrefacteurs ,"Inertie :", val_p_mod1[nombrefacteurs-1],
##              "/", matNorm.shape[1], "->",
##              np.round(val_p_mod1[nombrefacteurs-1] / matNorm.shape[1] * 100,2)
##              , "%", "Inertie subtotal", sommeInertiePercentual)
#        sommeInertiePercentual = sommeInertiePercentual \
#            + np.round(val_p_mod1[nombrefacteurs-1] / matNorm.shape[1] * 100,2)
#        if(flagBreak == 1):
#            break
#        if(sommeInertiePercentual > pourcentageInertie):
##            print("sommeInertiePercentual > pourcentageInertie")
##            print(sommeInertiePercentual, ">", pourcentageInertie)
#            flagBreak = 1
            
    print("Nombre de facteurs retenu = ", nombrefacteurs)
    
    print("_"*82)
    print("Classification ascendante hiérarchique (CAH)")
    print("Facteur modalité 1")        
    cah(fact_mod1[:,0:nombrefacteurs], noms_individus, noms_modalites)
    
    print("_"*82)
    print("Classification ascendante hiérarchique (CAH)")
    print("Facteur modalité 2")
    cah(fact_mod2[:,0:nombrefacteurs], noms_modalites, noms_individus)