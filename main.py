from PIL import Image, ImageOps                                   
import numpy as np
import scipy
import scipy.stats as st
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import scipy.misc

chat_img = Image.open("images/chat.png")
chat_img2 = Image.open("images/chat2.png")
chat_np = np.array(chat_img)
chat_np2 = np.array(chat_img2)

a = np.array([
    [[1, 2, 3], [1, 2, 3], [1, 2, 3]], 
    [[4, 5, 6], [4, 5, 6], [4, 5, 6]], 
    [[4, 5, 6], [4, 5, 6], [4, 5, 6]]
    ])

b = np.array([
    [ 10, 20, 30, 40, 60 ],
    [ 10, 20, 30, 40, 60 ],
    [ 10, 20, 30, 40, 60 ],
    [ 10, 20, 30, 40, 60 ]
])

# Générer un noyau Gaussien
def gkern(kernlen, nsig):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

# Supposé : la taille de l'image est au moins égale à celle du noyau (les noyaux sont en général petis comme 3x3)
# Technique pour les bords de l'image : étendre l'image au niveau des bords pour avoir des 0
# greyscale : True si l'image traitée est en noir et blanc / False sinon
# verbose : pour le debug
def convolulte(im, ker, greyscale=False, verbose=False):
    n = len(im)
    m = len(im[0])
    ker_size = len(ker)
    pxl_size = ( len(im[0, 0]) if not greyscale else 1 )

    if verbose:
        print('Le tableau sans les zeros :')
        print(im)

    im = fill_edges_zeros(im, ker_size, pxl_size)
    res = ( np.full((n, m, pxl_size), 0) if not greyscale else np.full((n, m), 0) )

    if verbose:
        print('Taille :', n, m)
        print('Le tableau vide :')
        print(res)
        print('Le tableau avec les zeros :')
        print(im)

    for i in range(0 , n):
        for j in range(0, m):

            # scope :
            # le tableau de pixels de taille ker_size x ker_size traité ensuite convoluter
            scope = ( np.zeros((ker_size, ker_size, pxl_size)) if not greyscale else np.zeros((ker_size, ker_size)) )

            for k in range(ker_size):
                c_range = [j , j + ker_size]
                #print(im[i-ker_size//2 + k], c_range, im[i-ker_size//2 + k][ c_range[0] : c_range[1] ])
                scope[k] = im[i + k][ c_range[0] : c_range[1] ]

            # On modifie les pixels du tableau de retour
            res[i, j] = prod_conv(np.array(scope), ker, greyscale=True, verbose=verbose)

    return res


# Permet de rajouter des zeros tout autour de l'image                
def fill_edges_zeros(arr, ker_size, pxl_size):
    n = len(arr)
    m = len(arr[0])
    p = (ker_size - 1) // 2

    for k in range(p):
        # Mettre des zeros sur la première colonne
        arr = np.insert(arr, 0, 0, axis = 1)

        # Mettre des zeros sur la deuxième colonne
        arr = np.insert(arr, len(arr[0]), 0, axis = 1)

    q = len(arr[0])
    for k in range(p):
        # Ajouter une rangée de zero au début de la matrice
        arr = np.append(arr, ([[[0]*pxl_size] * q] if pxl_size > 1 else [[0]* q]), axis=0)
        # Ajouter une rangée de zero à la fin de la matrice
        arr = np.insert(arr, 0, ([[[0]*pxl_size] * q] if pxl_size > 1 else [[0]* q]), axis=0)

    return arr

# Réaliser le produit de convolution entre le noyau et la zone passée en paramètre
# Renvoi le pixel central du produit

# le paramètre 'lum' à permet de :
#           - True : laisser inchangé la luminosité (4eme composante des fichiers PNG)
#           - False : convoluter la luminosité
# Par défaut à True
def prod_conv(arr, ker, lum=True, greyscale=False, verbose=False):

    if verbose:
        print(arr)

    if greyscale: lum = False

    n = len(ker)

    pxl_size = ( len(arr[0][0]) if not greyscale else 1 )
    res = [0] * pxl_size

    for i in range(n):
        for j in range(n):
            for k in range(pxl_size):
                res[k] += (( arr[i, j][k] if not greyscale else arr[i, j] ) * ker[i, j])

    if lum == True:
        res[len(res) - 1] = arr[n//2, n//2][len(arr[n//2, n//2]) - 1]

    if greyscale: res = res[0]

    if verbose: 
        print(res)

    return res


# Filtre de Canny : permet de faire une détection de bords
# Etapes :
# 1) utiliser un filtre gaussien pour réduire le bruit
# 2) Utiliser deux gradients (un selon Ox et un selon Oy) pour faire un détection des bords selon les deux axes
# 3) Combiner les deux gradients selon Ox et Oy pour obtenir un graident optimal obtenu selon la formule G = \sqrt{Gx^2 + Gy^2} (ensuite normalisé)
def canny_filter(im, threshold, output_name, verbose=False, output=False):

    arr = np.array(im)

    gker = gkern(3, 0.5)

    gx = np.array([
        [-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]])
    gy = np.array([
        [-1, -2, -1], 
        [0, 0, 0], 
        [1, 2, 1]])

    laplacian = np.array([
        [1,1,1],
        [1,-8,1],
        [1,1,1]])


    fgauss = convolulte(arr, gker, greyscale=True, verbose=verbose)

    fx = convolulte(fgauss, gx, greyscale=True, verbose=verbose)
    fy = convolulte(fgauss, gy, greyscale=True, verbose=verbose)

    grad = np.sqrt(np.square(fx) + np.square(fy))
    grad = grad / grad.max() * 255.0
    grad[grad<threshold] = 0

    lap = convolulte(grad, laplacian, greyscale=True, verbose=verbose)

    (M,N) = lap.shape

    lap_pad = np.zeros((M+2,N+2))
    lap_pad[1:-1,1:-1] = lap                                             
    result = np.zeros((M,N))

    # Trouver les "zeros-crossing" c'est à dire les bords à l'aide d'un filtre Laplacien
    for i in range(1,M+1):
        for j in range(1,N+1):
            if lap_pad[i,j]<0:
                for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                        if lap_pad[i+x,j+y]>0:
                            result[i-1,j-1] = 1

    A = detectCircles(result, 14, 20, radius=[50, 5])
    plotCircles(A)

    # plt.imshow(fx, cmap='gray')
    # plt.title('Selon x')
    # plt.show()

    # plt.imshow(fx, cmap='gray')
    # plt.title('Selon y')
    # plt.show()

    # plt.imshow(grad, cmap='gray')
    # plt.title('Resultat')
    # print(grad)
    # plt.show()

    if output:
        imout = Image.fromarray(grad).convert("L")
        imout.save('outputs/' + output_name + '_output.jpeg')


def gaussian_filter(im):
    arr = np.array(im)
    gker = gkern(9, 0.5)
    fgauss = convolulte(arr, gker, greyscale=False, verbose=False)

    plt.imshow(fgauss)
    plt.title('Resultat')
    plt.show()


def canny_test(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    G = np.array(G, dtype=np.uint8)
    print(G)
    
    plt.imshow(G, cmap='gray')
    plt.show()
    return Ix, Iy

def detectCircles(img,threshold,region=None, radius=None):

    (M,N) = img.shape

    if radius == None:
        radius = [max(M, N), 5]

    [R_max,R_min] = radius
    R = R_max - R_min

    # Tableau stockage des points de Hough : 3 dimensions => 1) Rayon   -  2 & 3) coordonnées X & Y
    # On ajoute un bord de 2*R_max pour eviter les dépassements de taille
    A = np.zeros((R_max, M+2*R_max, N+2*R_max))
    B = np.zeros((R_max, M+2*R_max, N+2*R_max))

    # Calculer tous les angles possibles (pour la calcul des equations de cercle ensuite)
    theta = np.arange(0,360)*np.pi/180

    # Coordonnées des points retenus (point=1)
    edges = np.argwhere(img[:,:])

    # On itère sur les valeurs de rayons
    for val in range(R):

        # Valeur du rayon
        r = R_min+val

        # Créer un modèle de cercle (tableau de zeros à la taille nécessaire)
        circle_temp = np.zeros((2*(r+1),2*(r+1)))

        # Centre du cercle dans notre modèle
        (m,n) = (r+1,r+1)

        for angle in theta:
            # Eqation de cercle
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))

            # Pour compléter notre modèle (1 pour les points du cercle, 0 sinon)
            circle_temp[m+x,n+y] = 1

        total = np.argwhere(circle_temp).shape[0]

        for x,y in edges:
            X = [x-m+R_max,x+m+R_max]
            Y= [y-n+R_max,y+n+R_max]
            A[r,X[0]:X[1],Y[0]:Y[1]] += circle_temp
        
        #print(A[r][A[r]<threshold*total/r].shape, A[r].shape)
        A[r][A[r]<threshold*total/r] = 0

    return A[:,R_max:-R_max,R_max:-R_max]
        
def plotCircles(A):
    img = plt.imread("images/circles.jpeg")
    fig, ax = plt.subplots()

    circleCoordinates = np.argwhere(A) 
    print(circleCoordinates)
    circle = []
    for r,x,y in circleCoordinates:
        ax.add_patch(plt.Circle((y,x),r,color='r',fill=False))
    ax.imshow(img)
    plt.show()


#print(fill_edges_zeros(np.random.rand(3,2), 6, 1))
canny_filter(ImageOps.grayscale(Image.open("images/circles.jpeg")), 30, 'lena', output=False, verbose=False)
#gaussian_filter(((Image.open("images/chat2.png"))))




