#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" I. Bloch
"""


#%% SECTION 1 inclusion of packages 


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import img_as_float
from skimage.segmentation import chan_vese
from skimage.segmentation import checkerboard_level_set
from skimage.segmentation import circle_level_set
from skimage.segmentation import morphological_geodesic_active_contour,  inverse_gaussian_gradient

import skimage.morphology as morpho  

#%%
def strel(forme,taille,angle=45):
    """renvoie un element structurant de forme  
     'diamond'  boule de la norme 1 fermee de rayon taille
     'disk'     boule de la norme 2 fermee de rayon taille
     'square'   carre de cote taille (il vaut mieux utiliser taille=impair)
     'line'     segment de langueur taille et d'orientation angle (entre 0 et 180 en degres)
      (Cette fonction n'est pas standard dans python)
    """

    if forme == 'diamond':
        return morpho.selem.diamond(taille)
    if forme == 'disk':
        return morpho.selem.disk(taille)
    if forme == 'square':
        return morpho.selem.square(taille)
    if forme == 'line':
        angle=int(-np.round(angle))
        angle=angle%180
        angle=np.float32(angle)/180.0*np.pi
        x=int(np.round(np.cos(angle)*taille))
        y=int(np.round(np.sin(angle)*taille))
        if x**2+y**2 == 0:
            if abs(np.cos(angle))>abs(np.sin(angle)):
                x=int(np.sign(np.cos(angle)))
                y=0
            else:
                y=int(np.sign(np.sin(angle)))
                x=0
        rr,cc=morpho.selem.draw.line(0,0,y,x)
        rr=rr-rr.min()
        cc=cc-cc.min()
        img=np.zeros((rr.max()+1,cc.max()+1) )
        img[rr,cc]=1
        return img
    raise RuntimeError('Erreur dans fonction strel: forme incomprise')

#%% SECTION 2 - Input image

im=skio.imread('coeurIRM.bmp')

#im=skio.imread('retineOA.bmp')

#im=skio.imread('brain.bmp')
#im=im[:,:,1]

#im=skio.imread('brain2.bmp')

plt.imshow(im, cmap="gray", vmin=0, vmax=255)

#%% SECTION 3a - Segmentation using active contours 

# faire meilleur init en réa utilise comme biomed landmark/ mask to init
# tester avec diffusion du gradient

s = np.linspace(0, 2*np.pi, 100)
r = 136 + 35*np.sin(s)
c = 129 + 35*np.cos(s)
init = np.array([r, c]).T

#alpha = contrainte de régularité de length => si elevé il faut regulariser bcp lenght
#beta = contrainte de regularité de courbature => si elevé plus courber
#alpha et beta sont un peu redondant quand on controle curbure on controle aussi longeur
#wedge = controle attraction to edge => si petit va pas etre attiré bcp par les gradients
#gamma = time step parameter => si trop grand loupe minima et trop petit trop longue convergence

#max_px_move => h dans formule ?
plt.imshow(im, cmap='gray')
plt.show()
gauss = gaussian(im, 0.1)
plt.imshow(gauss, cmap='gray')
plt.show()
snake = active_contour(gauss,
                       init, alpha=5, beta=5, w_edge=-20, gamma=0.002)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])

plt.show()

""" 2 ventricules
s = np.linspace(0, 2*np.pi, 100)
r = 133 + 40*np.sin(s)
c = 115 + 40*np.cos(s)
init = np.array([r, c]).T

#alpha = contrainte de régularité de length => si elevé il faut regulariser bcp lenght
#beta = contrainte de regularité de courbature => si elevé plus courber
#alpha et beta sont un peu redondant quand on controle curbure on controle aussi longeur
#wedge = controle attraction to edge => si petit va pas etre attiré bcp par les gradients
#gamma = time step parameter => si trop grand loupe minima et trop petit trop longue convergence

#max_px_move => h dans formule ?
plt.imshow(im, cmap='gray')
plt.show()
gauss = gaussian(im, 0.1)
plt.imshow(gauss, cmap='gray')
plt.show()
snake = active_contour(gauss,
                       init, alpha=5, beta=5, w_line=5, w_edge=-20, gamma=0.002)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])

plt.show()
"""
mask = np.zeros((im.shape))
for i in range(snake.shape[0]):
    x,y = snake[i]
    mask[int(x),int(y)] = 1
    

se=strel('disk',30)
mask=morpho.closing(mask,se)
plt.imshow(mask, cmap='gray')
plt.show()

plt.imshow(im*mask, cmap='gray')
plt.show()

#%% Avec balloon force
plt.imshow(im, cmap='gray')
plt.show()
gauss = gaussian(im, 0.1)
gimage = inverse_gaussian_gradient(gauss)
init_ls = np.zeros(im.shape, dtype=np.int8)
init_ls[130:140, 125:135] = 1
plt.imshow(init_ls, cmap='gray')
plt.show()
ls = morphological_geodesic_active_contour(gimage, iterations=300,
                                           init_level_set=init_ls,
                                           smoothing=1, balloon=1,
                                           threshold=0.8)
#ls = morphological_chan_vese(im, iterations=35, init_level_set=init_ls,
#                             smoothing=2, iter_callback=callback)

plt.imshow(ls, cmap='gray')
plt.show()
plt.imshow(ls*im, cmap='gray')
plt.show()

#%% SECTION 3b - Open contours

# Use retineOA.bmp

r = np.linspace(25, 80, 100)
c = np.linspace(17, 100, 100)
init = np.array([r, c]).T

plt.imshow(im, cmap='gray')
plt.show()
gauss = gaussian(im, 1)
plt.imshow(gauss, cmap='gray')
plt.show()

#bc fixed pour dire que c'est contour ouvert
#alpha = contrainte de régularité de length => si elevé il faut regulariser bcp lenght
#beta = contrainte de regularité de courbature => si elevé plus courber
#wline = attraction to brightness  (darkness if neg)
#wedge = controle attraction to edge => si petit va pas etre attiré bcp par les gradients
#gamma = time step parameter => si trop grand loupe minima et trop petit trop longue convergence

# active_contour => critere grad + courbe para
snake = active_contour(gauss, init, bc='fixed',
                       alpha=20, beta=1, w_line=-10, w_edge=20, gamma=0.001)

fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])

plt.show()

#%% brain 2
im = img_as_float(im)
image=im.copy()
plt.imshow(image, cmap='gray')
plt.show()


r = np.linspace(30, 240, 100)
c = np.linspace(130, 130, 100)
init = np.array([r, c]).T

plt.imshow(im, cmap='gray')
plt.show()
gauss = gaussian(im, 1)
plt.imshow(gauss, cmap='gray')
plt.show()

#bc fixed pour dire que c'est contour ouvert
#alpha = contrainte de régularité de length => si elevé il faut regulariser bcp lenght
#beta = contrainte de regularité de courbature => si elevé plus courber
#wline = attraction to brightness  (darkness if neg)
#wedge = controle attraction to edge => si petit va pas etre attiré bcp par les gradients
#gamma = time step parameter => si trop grand loupe minima et trop petit trop longue convergence

# active_contour => critere grad + courbe para
snake = active_contour(gauss, init, bc='fixed',
                       alpha=0.3, beta=0, w_line=-1, w_edge=30, gamma=0.001)

fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(im, cmap=plt.cm.gray)
#ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3, alpha=0.2)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])

plt.show()


mask = np.zeros((im.shape))

for i in range(snake.shape[0]):
    x,y = snake[i]
    mask[int(x), int(y):im.shape[1]] = 1
plt.imshow(mask,cmap='gray')
plt.show()
se=strel('disk',10)
mask=morpho.closing(mask,se)
plt.imshow(mask, cmap='gray')
plt.show()

plt.imshow(im*mask, cmap='gray')
plt.show()
plt.imshow(im*(1-mask), cmap='gray')
plt.show()


#%% SECTION 4 - Segmentation using level sets (and region homogeneity)

im = img_as_float(im)
image=im.copy()

N=15
for k in range(N):
    se=strel('disk',k)
    image=morpho.closing(morpho.opening(image,se),se)
plt.imshow(image, cmap='gray')
plt.show()

# Init avec un damier
#init_ls = checkerboard_level_set(image.shape, 6)

# Init avec un cercle
init_ls = circle_level_set(image.shape, (130,120), 10)

# Init avec plusieurs cercles
"""circleNum = 8
circleRadius = image.shape[0] / (3*circleNum)
circleStep0 = image.shape[0]/(circleNum+1)
circleStep1 = image.shape[1]/(circleNum+1)
init_ls = np.zeros(image.shape)
for i in range(circleNum):
        for j in range(circleNum):
            init_ls = init_ls + circle_level_set (image.shape, 
                                                  ((i+1)*circleStep0, (j+1)*circleStep1), circleRadius)

"""
plt.imshow(init_ls, cmap='gray')
plt.show()

# chan_vese => critere homo reg

#mu = edges length para smaller => smaller obj
#lambda1 = si < lamda2 region will have a larger range of values than the other.
#lambda2 = si < lamda1 region will have a larger range of values than the other.
#av value inside and outside circle => g
#each reg homo => evolue pour min energie
# rendre plus homogene intial level map avec original image...


#tol = Level set variation tolerance between iterations 
# si lambda1 fort on accepte pas une région pas homogene donc seg reste petit
# si lamnda1 moins fort on accepte que la région soit moins homogène donc on ajoute a la seg les regions prochent mais qui different un peu
# plus lambda2 fort plus région doit etre homogene a l'ext
cv = chan_vese(image, mu=0.25, lambda1=3, lambda2=1, tol=1e-3, max_iter=200,
               dt=0.5, init_level_set=init_ls, extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()

plt.imshow(cv[0], cmap='gray')
plt.show()
plt.imshow(im, cmap='gray')
plt.show()
plt.imshow(im*cv[0], cmap='gray')
plt.show()

#%% ventricule gauche

im = img_as_float(im)
image=im.copy()
se=strel('disk',2)
image=morpho.dilation(image,se)
plt.imshow(image, cmap='gray')
plt.show()

# Init avec un damier
#init_ls = checkerboard_level_set(image.shape, 6)

# Init avec un cercle
init_ls = circle_level_set(image.shape, (135,125), 10)

# Init avec plusieurs cercles
"""circleNum = 8
circleRadius = image.shape[0] / (3*circleNum)
circleStep0 = image.shape[0]/(circleNum+1)
circleStep1 = image.shape[1]/(circleNum+1)
init_ls = np.zeros(image.shape)
for i in range(circleNum):
        for j in range(circleNum):
            init_ls = init_ls + circle_level_set (image.shape, 
                                                  ((i+1)*circleStep0, (j+1)*circleStep1), circleRadius)

"""
plt.imshow(init_ls, cmap='gray')
plt.show()

# chan_vese => critere homo reg

#mu = edges length para smaller => smaller obj
#lambda1 = si < lamda2 region will have a larger range of values than the other.
#lambda2 = si < lamda1 region will have a larger range of values than the other.
#av value inside and outside circle => g
#each reg homo => evolue pour min energie
# rendre plus homogene intial level map avec original image...


#tol = Level set variation tolerance between iterations 
# si lambda1 fort on accepte pas une région pas homogene donc seg reste petit
# si lamnda1 moins fort on accepte que la région soit moins homogène donc on ajoute a la seg les regions prochent mais qui different un peu
# plus lambda2 fort plus région doit etre homogene a l'ext
cv = chan_vese(image, mu=0.25, lambda1=11, lambda2=1, tol=1e-3, max_iter=200,
               dt=0.5, init_level_set=init_ls, extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()

plt.imshow(cv[0], cmap='gray')
plt.show()
plt.imshow(im, cmap='gray')
plt.show()
plt.imshow(im*cv[0], cmap='gray')
plt.show()

#%% Brain

im = img_as_float(im)
image=im.copy()

N=10
for k in range(N):
    se=strel('disk',k)
    open=morpho.opening(image,se)
    recopen = morpho.reconstruction(open, image)
    clos=morpho.closing(recopen,se)
    image = morpho.reconstruction(clos, recopen, 'erosion')
plt.imshow(image,cmap="gray")
plt.show()


# Init avec un damier
#init_ls = checkerboard_level_set(image.shape, 6)

# Init avec un cercle
init_ls = circle_level_set(image.shape, (210,220), 10)

# Init avec plusieurs cercles
"""circleNum = 8
circleRadius = image.shape[0] / (3*circleNum)
circleStep0 = image.shape[0]/(circleNum+1)
circleStep1 = image.shape[1]/(circleNum+1)
init_ls = np.zeros(image.shape)
for i in range(circleNum):
        for j in range(circleNum):
            init_ls = init_ls + circle_level_set (image.shape, 
                                                  ((i+1)*circleStep0, (j+1)*circleStep1), circleRadius)

"""
plt.imshow(init_ls, cmap='gray')
plt.show()

# chan_vese => critere homo reg

#mu = edges length para smaller => smaller obj
#lambda1 = si < lamda2 region will have a larger range of values than the other.
#lambda2 = si < lamda1 region will have a larger range of values than the other.
#av value inside and outside circle => g
#each reg homo => evolue pour min energie
# rendre plus homogene intial level map avec original image...


#tol = Level set variation tolerance between iterations 
# si lambda1 fort on accepte pas une région pas homogene donc seg reste petit
# si lamnda1 moins fort on accepte que la région soit moins homogène donc on ajoute a la seg les regions prochent mais qui different un peu
# plus lambda2 fort plus région doit etre homogene a l'ext
cv = chan_vese(image, mu=0.25, lambda1=5, lambda2=1, tol=1e-3, max_iter=100,
               dt=0.5, init_level_set=init_ls, extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()

plt.imshow(cv[0], cmap='gray')
plt.show()
plt.imshow(im, cmap='gray')
plt.show()
plt.imshow(im*cv[0], cmap='gray')
plt.show()


#%% 3D image
from mpl_toolkits.mplot3d import Axes3D

X,Y = np.ogrid[0:cv[1].shape[0], 0:cv[1].shape[1]]
Z = cv[1]
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
plt.show()
"""
for i in range(cv[1].shape[0]):
    for j in range(cv[1].shape[1]):
        if cv[1][i,j] > 0:
            print(cv[1][i,j])
"""            
plt.imshow(np.where(cv[1] > 0, cv[1],0), cmap='gray')
plt.show()

#%% END  TP - Deformable Models

#%% Test
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


# Morphological ACWE
image = img_as_float(data.camera())

# Initial level set
init_ls = checkerboard_level_set(image.shape, 6)
# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)
ls = morphological_chan_vese(image, iterations=35, init_level_set=init_ls,
                             smoothing=3, iter_callback=callback)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors='r')
ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

ax[1].imshow(ls, cmap="gray")
ax[1].set_axis_off()
contour = ax[1].contour(evolution[2], [0.5], colors='g')
contour.collections[0].set_label("Iteration 2")
contour = ax[1].contour(evolution[7], [0.5], colors='y')
contour.collections[0].set_label("Iteration 7")
contour = ax[1].contour(evolution[-1], [0.5], colors='r')
contour.collections[0].set_label("Iteration 35")
ax[1].legend(loc="upper right")
title = "Morphological ACWE evolution"
ax[1].set_title(title, fontsize=12)


# Morphological GAC
image = img_as_float(data.coins())
gimage = inverse_gaussian_gradient(image)

# Initial level set
init_ls = np.zeros(image.shape, dtype=np.int8)
init_ls[10:-10, 10:-10] = 1
# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)

ls = morphological_geodesic_active_contour(gimage, iterations=230,
                                           init_level_set=init_ls,
                                           smoothing=1, balloon=-1,
                                           threshold=0.69,
                                           iter_callback=callback)

ax[2].imshow(image, cmap="gray")
ax[2].set_axis_off()
ax[2].contour(ls, [0.5], colors='r')
ax[2].set_title("Morphological GAC segmentation", fontsize=12)

ax[3].imshow(ls, cmap="gray")
ax[3].set_axis_off()
contour = ax[3].contour(evolution[0], [0.5], colors='g')
contour.collections[0].set_label("Iteration 0")
contour = ax[3].contour(evolution[100], [0.5], colors='y')
contour.collections[0].set_label("Iteration 100")
contour = ax[3].contour(evolution[-1], [0.5], colors='r')
contour.collections[0].set_label("Iteration 230")
ax[3].legend(loc="upper right")
title = "Morphological GAC evolution"
ax[3].set_title(title, fontsize=12)

fig.tight_layout()
plt.show()
