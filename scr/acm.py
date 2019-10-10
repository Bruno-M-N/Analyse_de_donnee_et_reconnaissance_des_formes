# -*- coding: utf-8 -*-
"""
UE INF a 4-EG : Analyse de données et reconnaissance des formes
@author: Bruno Moreira Nabinger
@author: Clément Vinot

Analyse factorielle des correspondances multiples - ACM 
"""


# https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulation
import numpy as np
import codage
import math
import matplotlib.pyplot as plt

def acm(mat_codage_condense, noms_individus, noms_variables):
    ''' Analyse factorielle des correspondances multiples 
    
        Réalise une Analyse factorielle des correspondances multiples (ACP) en
        affichant le Diagramme des inerties, la Projection des individus dans 
        le Premier plan factoriel et aussi la Projection du nuage des variables
        dans le Premier plan factoriel
    
        Parameters:
            X : matrice centrée-réduite calculée à partir des données brutes
            noms_individus (list de strings) : noms des individus pour la ACP
            noms_variables (list de strings) : noms des variables pour la ACP
        Returns:
            None
    '''
    
    # Construction du tableau disjonctif complet (TDC)
    # Dans le TDC, les lignes représentent les individus et les
# colonnes représentent les modalités des variables. Slide 182/261 
    

    nb_modalites_par_var = mat_codage_condense.max(0)
    nb_modalites = int(nb_modalites_par_var.sum()) #mise en clase

    XTDC = np.zeros((mat_codage_condense.shape[0], nb_modalites))
    for i in range(mat_codage_condense.shape[0]):
        for j in range(mat_codage_condense.shape[1]):
            XTDC[i, int(nb_modalites_par_var[:j].sum() \
                 + mat_codage_condense[i,j]) - 1] = 1

    noms_modalites = []
    for i in range(mat_codage_condense.shape[1]):
        for j in range(int(nb_modalites_par_var[i])):
            noms_modalites.append(noms_variables[i] + str(j+1))

    # Calcul des matrices X, M et D
    #- X : matrice calculée à partir du TDC
    #- D : matrice des poids des individus
    #- M : matrice des poids des modalités

    Xfreq = XTDC/ XTDC.sum()

    marge_ligne = Xfreq.sum(0).reshape(1, Xfreq.shape[1])
    marge_colonne = Xfreq.sum(1).reshape(Xfreq.shape[0], 1)

    Xindep = marge_ligne * marge_colonne

    X = Xfreq/Xindep - 1

    M = np.diag(marge_ligne[0,:])
    # print("M", M.shape)

#    #D : matrice des poids des profils-lignes : matrice diagonal de 
#    # coefficient fj
#    D = np.eye(X.shape[0])
#    for i in range(freq_relative.shape[0]):
#        D[i,i] = fi[i]

    D = np.diag(marge_colonne[:,0])
    # print("D", D.shape)

    # Unlike in many matrix languages, the product operator * operates
        # elementwise in NumPy arrays. The matrix product can be performed 
        # using the @ operator (in python >=3.5) or the dot function or method
    # Matrice de covariance
    Xcov_ind = X.T.dot(D.dot(X.dot(M))) #slide 46/261

    #--------------------------------------------------------------------------
        #numpy.linalg.eig
    # Compute the eigenvalues and right eigenvectors of a square array.
    #
    # L valeur propre 
    # U vecteur propre : chaque col c'est un vecteur propre
    L, U = np.linalg.eig(Xcov_ind)

    #Reorder-------------------------------------------------------------------
    #print("Eigen values\n",L)
    ind = np.argsort(L)[::-1] # ordre decroissant
    
    val_p_mod1 = np.sort(L)[::-1]
    # on reordene, pour tous lignes, les colonnes en suivant la correspondance
    # entre l'ordre decroissant des valeurs propres
    
    vect_p_mod1 = U[:,ind]
    #--------------------------------------------------------------------------

    
    fact_mod1 = X.dot(M.dot(vect_p_mod1))
    
    #slide 48,53 /261
    # Relations entre les axes d'inertie et les facteurs des deux nuages 
    # slide 52/261
    fact_mod2 = X.T.dot(D.dot(fact_mod1)) / np.sqrt(val_p_mod1)    
    
    inerties = 100 * val_p_mod1 / val_p_mod1.sum()  
    
    
    
    
    #Contribuition-------------------------------------------------------------
    contribuition_indiv = np.zeros(fact_mod1.shape)
    for i in range(fact_mod1.shape[1]):
        f = fact_mod1[:,i]
        contribuition_indiv[:,i] = 100 * D.dot(f*f) / val_p_mod1[i] 
                                                    #f.T.dot(D.dot(f))
    
    print(contribuition_indiv[:,0])

    print(inerties)
    
    plt.close('all') # Close all figures window
    plt.figure(1)
    # plot points with cluster dependent colors
    plt.scatter(fact_mod1[:,0], fact_mod1[:,1])
    for label,x,y in zip(noms_individus,fact_mod1[:,0],fact_mod1[:,1]):
        plt.annotate(label,
                     xy=(x,y),
                     #xytext=(-5,5),
                     #textcoords='offset points',
                     #textcoords='figure points',
                     ha='right', va='bottom',
                     #arrowprops=dict(arrowstyle = '->',
                     #    connectionstyle='arc3,rad=0')
                     )
    
    plt.axvline(linewidth=0.5, color = 'k')
    plt.axhline(linewidth=0.5, color = 'k')
    plt.title('ACM - Projection des individus')
       
    plt.figure(2)
    # plot points with cluster dependent colors
    plt.scatter(fact_mod2[:,0], fact_mod2[:,1])
    for label,x,y in zip(noms_modalites, fact_mod2[:,0],fact_mod2[:,1]):
        plt.annotate(label,
                     xy=(x,y),
                     #xytext=(-5,5),
                     #textcoords='offset points',
                     #textcoords='figure points',
                     ha='right', va='bottom',
                     #arrowprops=dict(arrowstyle = '->',
                     #                 connectionstyle='arc3,rad=0')
                     )
    
    plt.axvline(linewidth=0.5, color = 'k')
    plt.axhline(linewidth=0.5, color = 'k')
    plt.title('ACM - Projection des modalites')
    print("val_p_ind",val_p_mod1)
    print("fact_mod1",fact_mod1)
    print("fact_mod2",fact_mod2)
    return val_p_mod1, fact_mod1, fact_mod2
    

if __name__ == "__main__":

    # Lecture des données
    # Codes de CSP : 22
#    code_csp = readfile("TD3-donnees/csp-noms_modalites1.txt")
#    # Codes des disciplines : 6 
#    code_disciplines = readfile("TD3-donnees/csp-noms_modalites2.txt")
#
#    mat = np.loadtxt("TD3-donnees/csp-donnees.txt")
#    afc(mat,code_csp, code_disciplines)
    
    # Lecture des données 
    # Noms individus : 57
    noms_individus = readfile("TD3-donnees/pommes-noms_individus.txt")
    # Noms des variables : 9
    noms_variables = readfile("TD3-donnees/pommes-noms_variables.txt")
    # Codage condensé (Tableau de type Individus x Variables comme en ACP)
    mat = np.loadtxt("TD3-donnees/pommes-donnees.txt")
    acm(mat, noms_individus, noms_variables)