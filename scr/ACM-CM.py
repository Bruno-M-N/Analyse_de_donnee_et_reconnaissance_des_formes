# https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
import matplotlib.cm as cm


def readfile(filename):
    with open(filename) as f:
        data = f.readlines()
    return data

def normalisation(M):
    """
    Centre réduit la matrice prise en entrée.
    """
    var=M.std(0)
    mean=M.mean(0)
    M = (M-mean)/var
    
    return(M)

def quantitatif_en_qualitatif1(M,nb_classes):
    #intervalles égaux
    nb_lignes=len(M)
    nb_colonnes=len(M[0])
    
    tailles_int = (np.amax(M,0) - np.amin(M,0))/nb_classes
    
    Q=np.copy(M)
    
    for i in range(nb_lignes) :
        for j in range(nb_colonnes):
            
            if Q[i][j] == np.amax(M,0)[j] :
                Q[i][j] = nb_classes
            
            else :
                Q[i][j] = int(np.floor((Q[i][j]-np.amin(M,0)[j])/tailles_int[j])+1)
    
    return Q

def acm(X,noms_variables):
    ''' Analyse factorielle des correspondances multiples 
    
        Réalise une Analyse factorielle des correspondances multiples (AFC)
        Renvoie la matrice des coordonnées sur les deux principaux axes factoriels.
    
        Paramètres:
            X : matrice centrée-réduite calculée à partir des données brutes
            noms_variables (list de strings) : noms des variables pour la AFC
        Retourne:
            fact_mod_1, fact_mod_2, noms_modalites
            respectivement, les coordonnés des deux nuages de points projetés sur les 
            deux axes factoriels principaux ; les noms des modalités.
    '''
    
    #Mise en classe des données quantitatives en vue d'y appliquer une ACM
    nb_modalites_par_var = X.max(0)
    nb_modalites = int(nb_modalites_par_var.sum())
    
    XTDC = np.zeros((X.shape[0],nb_modalites))
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            XTDC[i,int(nb_modalites_par_var[:j].sum()+X[i,j])-1] = 1
    
    #attribution des noms de modalités à partir des noms de variables, et de leur indice.
    noms_modalites = []
    for i in range(X.shape[1]):
        for j in range(int(nb_modalites_par_var[i])):
            noms_modalites.append(noms_variables[i]+str(j+1))
    
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

#    #D : matrice des poids des profils-lignes : matrice diagonal de coefficient fj
#    D = np.eye(X.shape[0])

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
    # U vecteur propre : chaque colonne est un vecteur propre
    L, U = np.linalg.eig(Xcov_ind)

    #Reorder-------------------------------------------------------------------
    #print("Eigen values\n",L)
    ind = np.argsort(L)[::-1] # ordre decroissant
    
    val_p_mod1 = np.sort(L)[::-1]
    # on reordene, pour tous lignes, les colonnes en suivant la correspondance
    # entre l'ordre decroissant des valeurs propres
    vect_p_mod1 = U[:,ind]
    #--------------------------------------------------------------------------
    #calcul des facteurs pour les deux nuages.
    fact_mod1 = X.dot(M.dot(vect_p_mod1))
    fact_mod2 = X.T.dot(D.dot(fact_mod1)) / np.sqrt(val_p_mod1)
    
    #Je conserve uniquement les deux premiers axes factoriels maximisant l'inertie
    fact_mod1 = fact_mod1[:,[0,1]]
    fact_mod2 = fact_mod2[:,[0,1]]
    
    return(fact_mod1,fact_mod2,noms_modalites)

def kmoyennes(X, k):
    """
    Réalise une classification par placement de centres mobiles (CM)
    
    Paramètres:
        X : Matrice de données qualitatives.
        k : Nombre de classes à réaliser (et donc de centre à placer).
        
    Retourne :
        centroids : le tableau des coordonnées des centres.
        labels : le tableau du centre le plus proche pour chaque point.
        
    """
    centroids, labels = kmeans2(np.real(X),k,iter=100)
    #centroids : le tableau des coordonnées des centres.
    #labels : le tableau du centre le plus proche par point.
    
    return centroids,labels
    
if __name__ == "__main__":
    """ Execution du programme pour un set de données
    
    Execute pour un set de données, la normalisation des données, puis l'acm, la CM,
    et enfin trace le nuage de point en faisant apparaitre les différentes classes.
    """
    # Lecture des données
    data = np.loadtxt("TD3-donnees/population_donnees.txt")
    # Codes de CSP : 22
    noms_individus = readfile('TD3-donnees/population_noms_individus.txt')
    # Codes des disciplines : 6 
    noms_variables = readfile('TD3-donnees/population_noms_variables.txt')
    
    X = normalisation(data)
        
    #Choix de réaliser k classes arbitrairement
    k=3
    
    #On réalise une mise en classe avant d'appliquer l'ACM
    X = quantitatif_en_qualitatif1(X,k)
    
    Y,Z,noms_modalites = acm(X, noms_variables)
    #Réalisation d'une acp des données et récupération de la projection du nuage des individus
    #et des variables
    
    #Tracé du nuage des individus en prenant une couleur différente pour chaque classe.
    centroids, labels = kmoyennes(Y,k)
    colors = cm.rainbow(np.linspace(0, 1, k))
    
    fig = plt.figure(figsize=[10,10])
    for i in range(k):
        plt.scatter(centroids[i][0],centroids[i][1],c=colors[i],marker='*')
    
    for i in range(Y.shape[0]):
        color = colors[labels[i]]
        plt.scatter(Y[i][0],Y[i][1], c=color)
        plt.annotate(noms_individus[i],(Y[i][0],Y[i][1]))
    plt.title('nuage des individus après ACM et CM')
    
    #Tracé du résultat en prenant une couleur différente pour chaque classe.    
    centroids, labels = kmoyennes(Z,k)
    fig = plt.figure(figsize=[10,10])
    for i in range(k):
        plt.scatter(centroids[i][0],centroids[i][1],c=colors[i],marker='*')
    
    for i in range(Z.shape[0]):
        color = colors[labels[i]]
        plt.scatter(Z[i][0],Z[i][1], c=color)
        plt.annotate(noms_modalites[i],(Z[i][0],Z[i][1]))
    plt.title('nuage des variables après ACM et CM')