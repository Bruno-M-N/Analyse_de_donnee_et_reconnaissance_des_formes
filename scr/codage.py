# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 08:10:28 2019

@author: User
"""

# https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulation
import numpy as np

def readfile(filename):
    with open(filename) as f:
        data = f.readlines()
    return data

def basicStatics(mat):
    '''Étude univariée qui consiste à décrire individuellement chaque variable
    afin de mieux les connaître. Pour chaque variable du tableau de données, la
    moyenne, l'écart-type, la variance,la valeur mininale, la valeur maximale, 
    et l'étendue sont calculés
    '''
    #print("Moyenne, Écart-type, Variance, Min, Max, Étendue")
    #La moyenne
    meanCol = mat.mean(0)
    #L'écart-type
    stdCol = mat.std(0)
    #La variance
    varCol = stdCol**2
    #La valeur minimale
    minCol = mat.min(0)
    #La valeur maximale
    maxCol = mat.max(0)
    #l’étendue
    amplitudeCol = maxCol - minCol 
    print("Moyenne   ", np.array2string(meanCol, formatter={'float_kind':
                              lambda meanCol: "%6.1f" % meanCol}))
    print("Écart-type", np.array2string(stdCol, formatter={'float_kind':
                              lambda stdCol: "%6.1f" % stdCol}))
    print("Variance  ", np.array2string(varCol, formatter={'float_kind':
                              lambda varCol: "%6.1f" % varCol}))
    print("Minimum   ", np.array2string(minCol, formatter={'float_kind':
                              lambda minCol: "%6.1f" % minCol}))
    print("Maximum   ", np.array2string(maxCol, formatter={'float_kind':
                              lambda maxCol: "%6.1f" % maxCol}))
    print("Étendue   ", np.array2string(amplitudeCol, formatter={'float_kind':
                              lambda amplitudeCol: "%6.1f" % amplitudeCol}))
    #print("Moyenne", np.array2string(meanCol, precision=2, separator='.'))

def normalisation(mat):
    '''Prend en argument une matrice correspondant à un tableau croisant des 
    individus en ligne et des variables quantitatives en colonnes. Le tableau 
    centré-réduit sera alors renvoyé.
    '''
    # Returns the average of the array elements. The average is taken over the 
    # flattened array by default, otherwise over the specified axis.
    meanCol = mat.mean(0)
    stdCol = mat.std(0)
    matNorm = (mat - meanCol)/stdCol #divison terme a terme
    return matNorm

def quantitatif_en_qualitatif1(mat, nClasses):
    '''Prennent en argument une matrice correspondant à un tableau croisant des 
    individus en ligne et des variables quantitatives en colonnes. Une mise en 
    classes sera réalisée en utilisant un découpage des variables quantitatives
    en intervalles égaux . Le nombre d’intervalles est donné en deuxième 
    argument de la fonction. 
    Le tableau croisant des individus avec des variables qualitatives sera 
    renvoyé, la valeur de ces variables pour un individu donné étant l’indice 
    de l’intervalle contenant la valeur de la variable quantitative d’origine.
    '''

    matMinFromCol = mat.min(0)
    matMaxFromCol = mat.max(0)
    lenghtIntervale = (matMaxFromCol - matMinFromCol)/nClasses
    print("Lenght Intervale: ", lenghtIntervale)
    
    bornes = matMinFromCol
    for i in range (1, nClasses):
        bornes = np.vstack((bornes, matMinFromCol + i * lenghtIntervale))

    #numpy.org/devdocs/reference/generated/numpy.vstack.html#numpy.vstack
    bornes = np.vstack((bornes, matMaxFromCol))
#    print("matMinFromCol ", matMinFromCol)
#    print("matMaxFromCol ", matMaxFromCol)
#    print("bornes", bornes)
    for i in range(bornes.shape[0]):
        ligne = bornes[i,:]
        print("Bornes    ", np.array2string(ligne, formatter={'float_kind':
                              lambda ligne: "%6.1f" % ligne}))
    
    mqual = np.zeros(mat.shape, dtype='int')
    
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            k = 0
            find = False
            while not find:
                if mat[i,j] >= bornes[k,j] and mat[i,j] <= bornes[k+1,j]:
                    find = True
                    mqual[i,j] = k + 1
                k = k + 1
#    print("mqual \n", mqual)            
    return mqual
    
def quantitatif_en_qualitatif2(mat, nClasses):
    '''Prennent en argument une matrice correspondant à un tableau croisant des 
    individus en ligne et des variables quantitatives en colonnes. Une mise en 
    classes sera réalisée en utilisant un découpage des variables quantitatives
    en effectifs égaux pour la deuxième. Le nombre d’intervalles est donné en 
    deuxième argument de la fonction. 
    Le tableau croisant des individus avec des variables qualitatives sera 
    renvoyé, la valeur de ces variables pour un individu donné étant l’indice
    de l’intervalle contenant la valeur de la variable quantitative d’origine.
    '''

    #numpy.org/devdocs/reference/generated/numpy.argsort.html#numpy.argsort
    indices = np.argsort(mat, axis = 0)
    nOfMembers = mat.shape[0]//nClasses
    
    classes = np.zeros(mat.shape[0], dtype = 'int')
    
    indBegin = 0
    for i in range(nClasses):
        indEnd = indBegin + nOfMembers - 1
        classes[indBegin:indEnd] = i
        indBegin = indEnd + 1
        
    mqual = np.zeros(mat.shape, dtype = 'int')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            indices
            
    return np.vsplit(mat,nClasses)    

if __name__ == "__main__":
    mat = np.loadtxt("donnees/population_donnees.txt")
    print("Matrice des données", mat.shape,": \n")
    for i in range(mat.shape[0]):
        #print("           ", mat[i,:])
        ligne = mat[i,:]
        print("          ", np.array2string(ligne, formatter={'float_kind':
                              lambda ligne: "%6.1f" % ligne}))
    #matNorm = normalisation(mat)
    #print("Matice normalise", matNorm.shape,": \n" , matNorm)
    
    print("\n Découpage des variables quantitatives en intervalles égaux")
    nClasses = 5
    print("Nombre de Classes :", nClasses)
    quantitatif_en_qualitatif1(mat, nClasses)
#    
#    print("\n Découpage des variables quantitatives en effectifs égaux")
#    nClasses = 5
#    print("Nombre de Classes :", nClasses)
#    print(quantitatif_en_qualitatif2(mat, nClasses))
    