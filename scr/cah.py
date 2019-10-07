"""
UE INF a 4-EG : Analyse de données et reconnaissance des formes
@author: Bruno Moreira Nabinger
@author: Clément Vinot

Classification ascendante hiérarchique - CAH 
"""

# https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.vq import whiten

def readfile(filename):
    with open(filename) as f:
        data = f.readlines()
    return data

def cah(X, noms_lignes, noms_colonnes):
    ''' Classification ascendante hiérarchique
    
        RÃ©alise une Classification non supervisée du type ascendante 
        hiérarchique (CAH) en affichant une hiérarchie de partitions, le 
        dendrogrammes, et 
        
        Parameters:
            X : matrice centrée-réduite calculée à partir des données brutes
            noms_individus (list de strings) : noms des individus pour la ACP
            noms_variables (list de strings) : noms des variables pour la ACP
        Returns:
            None
    '''
    
    # Construction de l’arbre hiérarchique par agrégations successives de deux
    # éléments-----------------------------------------------------------------
    # scipy.cluster.hierarchy.linkage(y, method='single', metric='euclidean',
    #                                 optimal_ordering=False)[source]
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html 
    # Perform hierarchical/agglomerative clustering.
    Z = linkage(X, 'ward')

    # Affichage de l'arbre hiÃ©rarchique par agrÃ©gations successives grÃ¢ce Ã  
    # un dendrogram------------------------------------------------------------
    print("Dendrogram")
    plt.title('Hierarchical clustering dendrogram') 
    plt.xlabel('sample index')
    plt.ylabel ('distance')
    dendrogram(
    Z,
    # leaf_label_rotation = 30,# rotates the x axis labels
    leaf_font_size = 8, # font size for the x axis labels
    )

    # Coupure de l'arbre hiérarchique pour obtenir une partition---------------
    # scipy.cluster.hierarchy.fcluster(Z, t, criterion='inconsistent', depth=2,
    #                                  R=None, monocrit=None)[source]
    # Forms flat clusters from the hierarchical clustering defined by the 
    # linkage matrix Z.
    k = 3
    clusters = fcluster(Z, k, criterion = 'maxclust')
    print("Clusters")
    print(clusters)
    
    # Affichage de l'arbre hiérarchique par agrégations successives grâce à  
    # un dendrogram------------------------------------------------------------

    plt.figure(figsize=(10, 8))
    plt.scatter(X[:,0], X[:,1], c = clusters)#, cmap = plt.cm.spectral) 
                                             # plot points with cluster depende

    for label, x, y in zip(noms_lignes, X[:,0], X[:,1]):
        plt.annotate(label, 
        xy = (x,y), # The point (x,y) to annotate
        xytext = (-5,5), # The position (x,y) to place the text at
        textcoords = 'offset points', # Offset (in points) from the xy value
        ha = "right", va = "bottom",
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad=0')
        )

    plt.title('moyennes')
    plt.show()


if __name__ == "__main__":

    # Lecture des données------------------------------------------------------
    data = np.loadtxt("TD3-donnees/csp-donnees.txt")
    # Codes de CSP : 22
    noms_lignes = readfile('TD3-donnees/csp-noms_modalites1.txt')
    # Codes des disciplines : 6 
    noms_colonnes = readfile('TD3-donnees/csp-noms_modalites2.txt')

    cah(data, noms_lignes, noms_colonnes)