# https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
import matplotlib.cm as cm


def readfile(filename):
    with open(filename) as f:
        data = f.readlines()
    return data

#centre réduit la matrice
def normalisation(M):
    
    var=M.std(0)
    mean=M.mean(0)
    M = (M-mean)/var
    
    return(M)

#Renvoie Y, les facteurs des deux axes d'inerties maximum pour le nuage de points X
def acp(X):
    ''' Analyse en Composantes Principales
    
        Paramètres:
            X : matrice centrée-réduite calculée à partir des données brutes
            
        Retourne:
            Y,Z : matrices des coordonnées du nuage des individus resp. variables 
            projetées sur les deux axes factoriels principaux.
    '''
    
    #Calcul des matrices M,D
    M = np.eye(X.shape[1])
    D = np.eye(X.shape[0]) / X.shape[0]
    

    Xcov_ind = X.T.dot(D.dot(X.dot(M)))
    
    # L valeur propre 
    # U vecteur propre : chaque colonne est un vecteur propre
    L, U = np.linalg.eig(Xcov_ind)
    #Classification des axes factoriels par ordre de valeur propres décroissantes
    #Ce que correspond à l'ordre d'inertie décroissant.
    ind = np.argsort(L)[::-1]
    vect_p_ind = U[:,ind]
    val_p_ind = np.sort(L)[::-1]
    #Calcul des facteurs des individus, puis des variables
    fact_ind = X.dot(M.dot(vect_p_ind))
    fact_var = X.T.dot(D.dot(fact_ind)) / np.sqrt(val_p_ind)
    #je dispose maintenant des facteurs, ceux qui m'intéressent sont les deux premiers
    #Pour avoir la projection sur un plan.
    
    Y = fact_ind[:,[0,1]]
    Z = fact_var[:,[0,1]]
    
    return Y,Z




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
    
    Execute pour un set de données, la normalisation des données, puis l'acp, la CM,
    et enfin trace le nuage de point en faisant apparaitre les différentes classes.
    """
    
    # Lecture des données
    data = np.loadtxt("TD3-donnees/population_donnees.txt")
    # Codes de CSP : 22
    noms_individus = readfile('TD3-donnees/population_noms_individus.txt')
    # Codes des disciplines : 6 
    noms_variables = readfile('TD3-donnees/population_noms_variables.txt')
    
    #centrage réduction des données
    data = normalisation(data)
    
    #Réalisation d'une acp des données et récupération de la projection du nuage des individus
    #et des variables
    Y, Z = acp(data)
    
    #Classification des données
    
    #Choix de réaliser k classes arbitrairement
    k=5
    
    centroids, labels = kmoyennes(Y,k)
    
    #Tracé du résultat en prenant une couleur différente pour chaque classe.
    colors = cm.rainbow(np.linspace(0, 1, k))
    
    fig = plt.figure(figsize=[10,10])
    for i in range(k):
        plt.scatter(centroids[i][0],centroids[i][1],c=colors[i],marker='*')
    
    for i in range(Y.shape[0]):
        color = colors[labels[i]]
        plt.scatter(Y[i][0],Y[i][1], c=color, label = noms_individus[i])
        plt.annotate(noms_individus[i],(Y[i][0],Y[i][1]))
    plt.title('nuage des individus après ACP et CM')
    
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