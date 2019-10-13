# -*- coding: utf-8 -*-
"""
UE INF a 4-EG : Analyse de données et reconnaissance des formes
@author: Bruno Moreira Nabinger
@author: Clément Vinot

Méthode mixte : Analyse en Composantes Principales (ACP)
              + Classification ascendante hiérarchique (CAH) 
"""

# https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulation
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
    matNorm = normalisation(mat)
    
    print("Analyse en Composantes Principales (ACP)")
    val_p_ind, fact_ind, fact_var = acp(matNorm, noms_individus, 
                                        noms_variables)
    #
    print("Résultat de l'analyse factoriel")
    print("Nombre de facteurs = ", len(val_p_ind))
    
    sommeInertiePercentual = 0
    nombreFacteurs = 1
    nombreFacteursRetenus = 1
    flagBreak = 0
    pourcentageInertie = 95
    for nombreFacteurs in range(1,len(val_p_ind)):
        # Inertie = valeur propre / inertie totale 
        print(nombreFacteurs ,"Inertie :", val_p_ind[nombreFacteurs-1],
              "/", matNorm.shape[1], "->",
              np.round(val_p_ind[nombreFacteurs-1] / matNorm.shape[1] * 100,2)
              , "%", "Inertie subtotal", sommeInertiePercentual)
        sommeInertiePercentual = sommeInertiePercentual \
            + np.round(val_p_ind[nombreFacteurs-1] / matNorm.shape[1] * 100,2)
        if(flagBreak == 1):
            nombreFacteursRetenus = nombreFacteurs
            flagBreak = 0
#            break
        if(sommeInertiePercentual > pourcentageInertie):
#            print("sommeInertiePercentual > pourcentageInertie")
#            print(sommeInertiePercentual, ">", pourcentageInertie)
            flagBreak = 1
            
    print("Nombre de facteurs retenu = ", nombreFacteursRetenus)
    
    print("_"*82)
    print("Classification ascendante hiérarchique (CAH)")
    print("Facteur individus")        
    cah(fact_ind[:,0:nombreFacteursRetenus], noms_individus, noms_variables)
    
    print("_"*82)
    print("Classification ascendante hiérarchique (CAH)")
    print("Facteur variables")
    cah(fact_var[:,0:nombreFacteursRetenus], noms_variables, noms_individus)