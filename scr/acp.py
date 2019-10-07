# -*- coding: utf-8 -*-
"""
UE INF a 4-EG : Analyse de données et reconnaissance des formes
@author: Bruno Moreira Nabinger
@author: Clément Vinot

Analyse en Composantes Principales - ACP 
"""

# https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulation
import numpy as np
import codage
import math
import matplotlib.pyplot as plt

def basicStatics(mat):
    #print("Moyenne, Écart-type, Variance, Min, Max, Étendue")
    for j in range(mat.shape[1]):
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
        #print(meanCol, stdCol, varCol, minCol, maxCol, amplitudeCol)

def acp(X,noms_individus, noms_variables):
    ''' Analyse en Composantes Principales
    
        Réalise une Analyse en Composantes Principales (ACP) en affichant le 
        Diagramme des inerties, la Projection des individus dans le Premier 
        plan factoriel et aussi la Projection du nuage des variables dans le
        Premier plan factoriel
    
        Parameters:
            X : matrice centrée-réduite calculée à partir des données brutes
            noms_individus (list de strings) : noms des individus pour la ACP
            noms_variables (list de strings) : noms des variables pour la ACP
        Returns:
            None
    '''
    
    # X : matrice centrée-réduite calculée à partir des données brutes
    # Les coordonnées xik des I points du nuage N_I dans l’espace RK forment 
    # une matrice notée X
    M = np.eye(X.shape[1])
    # D : matrice des poids des individus (on considère dans cette étude que 
    # tous les individus jouent le même rôle)
    D = np.eye(X.shape[0]) / X.shape[0]
    
        # Unlike in many matrix languages, the product operator * operates
        # elementwise in NumPy arrays. The matrix product can be performed 
        # using the @ operator (in python >=3.5) or the dot function or method
    # Matrice de covariance
    Xcov_ind = X.T.dot(D.dot(X.dot(M))) #slide 46/261

    #numpy.linalg.eig
    # Compute the eigenvalues and right eigenvectors of a square array.
    #
    # L valeur propre 
    # U vecteur propre : chaque col c'est un vecteur propre
    L, U = np.linalg.eig(Xcov_ind)

    #Reorder-------------------------------------------------------------------
    #print("Eigen values\n",L)
    ind = np.argsort(L)[::-1] # ordre decroissant
    
    val_p_ind = np.sort(L)[::-1]
    # on reordene, pour tous lignes, les colonnes en suivant la correspondance
    # entre l'ordre decroissant des valeurs propres
    
    vect_p_ind = U[:,ind]
    #--------------------------------------------------------------------------

    
    fact_ind = X.dot(M.dot(vect_p_ind))
    
    #slide 48,53 /261
    # Relations entre les axes d'inertie et les facteurs des deux nuages 
    # slide 52/261
    fact_var = X.T.dot(D.dot(fact_ind)) / np.sqrt(val_p_ind)    
    
    inerties = 100 * val_p_ind / val_p_ind.sum()  
    
    #Verification--------------------------------------------------------------
#    Xcov_var = X.dot(M.dot((X.T).dot(D)))
#    L, U = np.linalg.eig(Xcov_var)
#    
#    print("Val p variables :")
#    print(np.sort(L)[::-1])
    #--------------------------------------------------------------------------
    
    #Contribuition des individus-----------------------------------------------
    contribuition_indiv = np.zeros(fact_ind.shape)
    for i in range(fact_ind.shape[1]):
        f = fact_ind[:,i]
        contribuition_indiv[:,i] = 100 * D.dot(f*f) / val_p_ind[i] 
    #                                                 f.T.dot(D.dot(f))
    print('Contribution de représentation des individus')
    print(contribuition_indiv[:,0])

    #Qualité de représentation-------------------------------------------------
    # vecteur collone de dim 28
    dist = (fact_ind**2).sum(1)
    tdist = dist.reshape(len(noms_individus),1)
    print(tdist.shape)
    qualite_ind = fact_ind**2 / tdist
    print('Qualité de représentation des individus')
    print(qualite_ind[:,0])

    #Inerties------------------------------------------------------------------
    print('Inerties')
    print(inerties)
    plt.figure(1)
    plt.plot(inerties,'o-')
    plt.title("Diagramme des inerties")
    plt.grid()
    plt.show()

    print((fact_ind[:,0].T.dot(D.dot(fact_ind[:,0]))))
    print("Premier valeur propre :", val_p_ind[0],"/", X.shape[1], "->",
          np.round(val_p_ind[0] / X.shape[1] * 100,2), "%")
    print("Deuxième valeur propre :", val_p_ind[1],"/", X.shape[1], "->",
          np.round(val_p_ind[1] / X.shape[1] * 100,2), "%")
    
    F1, F2 = fact_ind[:,0], fact_ind[:,1]
    print("F1", F1)
    print("F2", F2)

#    plt.figure(2)
#    axes = plt.gca()
#    plt.axis('on')
#    #axes.set(xlim=(-4.5, 7), ylim=(-4.5, 3))
#    axes.add_artist(plt.Line2D((-5, 7), (0, 0),
#                              color = 'black', linewidth = 2))
#    axes.add_artist(plt.Line2D((0, 0), (-4.5, 3),
#                              color = 'black', linewidth = 2))
#    axes.set(xlabel='F1', ylabel='F2',
#       title='Premier plan factoriel')
#    plt.plot(F1, F2,'o')
#    # https://www.saltycrane.com/blog/2007/12/iterating-through-two-lists-in-parallel/
#    # zip returns a list of tuples, where the i-th tuple contains the i-th 
#    # element from each of the argument sequences or iterables. 
#    # This is useful for iterating over two lists in parallel.
#    for label,x,y in zip(noms_individus, F1, F2):
#        #plt.text(x, y, label)
#        plt.annotate(label,
#                     xy = (x,y), # The point (x,y) to annotate
#                     # The position (x,y) to place the text at
#                     xytext = (-40,10), 
#                     # Offset (in points) from the xy value
#                     textcoords = 'offset points',        
#                     #textcoords='figure points',
#                     #ha='right', va='bottom',
#                     arrowprops=dict(arrowstyle = '->',
#                                     connectionstyle='arc3,rad=0')
#                     )
#    #for i in range(0,14):
#    #    plt.text(F1[i], F2[i], noms_individus[i])
#    plt.grid()
#    plt.show()
    
    G1, G2 = fact_var[:,0], fact_var[:,1]
    print("G1", G1)
    print("G2", G2)
    
    plt.figure(3)
    x = np.arange(-1,1,0.0001)
    cercle_unite = np.zeros((2,len(x)))
    cercle_unite[0,:] = np.sqrt(1-x**2)
    cercle_unite[1,:] = -cercle_unite[0,:]  
    plt.plot(x, cercle_unite[0,:])
    plt.plot(x, cercle_unite[1,:])
    axes = plt.gca()
    plt.axis('on')
    #axes.set(xlim=(0.75, 1.025), ylim=(-0.75, 0.75))
    axes.add_artist(plt.Line2D((-1.2, 1.2), (0, 0),
                              color = 'black', linewidth = 2))
    axes.add_artist(plt.Line2D((0, 0), (-1.25, 1.2),
                              color = 'black', linewidth = 2))
    axes.set(xlabel='G1', ylabel='G2',
       title='Premier plan factoriel : projection du nuage des variables')
    plt.plot(G1, G2,'o')
    
    # https://www.saltycrane.com/blog/2007/12/iterating-through-two-lists-in-parallel/
    # zip returns a list of tuples, where the i-th tuple contains the i-th 
    # element from each of the argument sequences or iterables. 
    # This is useful for iterating over two lists in parallel.
    for label,x,y in zip(noms_variables, G1, G2):
        plt.annotate(label,
                     xy = (x,y), # The point (x,y) to annotate
                     # The position (x,y) to place the text at
                     xytext = (-50,5), 
                     # Offset (in points) from the xy value
                     textcoords = 'offset points',
                     #textcoords='figure points',
                     #ha='right', va='bottom',
                     arrowprops=dict(arrowstyle = '->',
                                     connectionstyle='arc3,rad=0')
                     )
    #for i in range(0,12):
    #    plt.text(G1[i], G2[i], noms_variables[i])
    plt.grid()
    plt.show()


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
    acp(matNorm[:,0:10],noms_individus, noms_variables)
    
#    noms_individus = readfile("donnees/villes_noms_individus.txt")
#    noms_variables = readfile("donnees/villes_noms_variables.txt")
#
#    mat = np.loadtxt("donnees/villes_donnees.txt")
#    #print("Matrice", mat.shape,": \n" ,  mat)
#    basicStatics(mat)
#    matNorm = normalisation(mat)
#    acp(matNorm[:,0:12],noms_individus, noms_variables)
    