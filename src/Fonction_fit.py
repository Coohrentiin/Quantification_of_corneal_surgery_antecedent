# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:30:46 2019
Last update: Th Jul 23 2020 by Maëlle
Différentes fonctions de fit

@author: bocheux
"""
import numpy as np
import scipy.signal as sgn
import matplotlib.pyplot as plt


def fit_curve2D_seuil(image, seuil, w_filter, filter_order, sgolay_order):
    # Retourne les positions des max d'une image 2D : cette fonction a été conçue pour
    # le cas où les maximums d'intensité de la cornée ne correspondent pas toujours à l'épithelium
    # (qui est alors plus compliqué à déterminer). Elle utilise une version binarisée de l'image.
	image_seuil = image > seuil
	colonne = np.shape(image_seuil)[1]
	maxima = np.zeros(colonne)
	for i in range(colonne):
		ones_positions = np.where(image_seuil[:, i])[0]
		if ones_positions.shape == (0,):
			maxima[i] = 0
		else:
			maxima[i] = ones_positions[0]
	maxima_null_full = np.where(maxima==0)[0]
	if len(maxima_null_full) != 0:
		x_delta = 50
#        delim_bordure = image.shape[1]/2
		delim_bordure = 150
		if maxima_null_full[0] < delim_bordure: # à gauche
        		maxima_null = maxima_null_full[maxima_null_full < delim_bordure]
        		x1 = maxima_null[-1]+1
        		y1 = maxima[x1]
        		x2 = x1 + x_delta
        		y2 = maxima[x2]
        		a = (y2-y1)/(x2-x1)
        		b = y1 - a*x1
        		temp = np.linspace(a*maxima_null[0]+b,y1,maxima_null.shape[0])
        		temp = temp.astype(int)
        		maxima[maxima_null ] = temp
		if maxima_null_full[-1] > colonne - delim_bordure: # à droite    
        		maxima_null = maxima_null_full[maxima_null_full > colonne - delim_bordure]
        		x2 = maxima_null[0]-1
        		y2 = maxima[x2]
        		x1 = x2 - x_delta
        		y1 = maxima[x1]
        		a = (y2-y1)/(x2-x1)
        		b = y1 - a*x1
        		temp = np.linspace(y2,a*maxima_null[-1]+b,maxima_null.shape[0])
        		temp = temp.astype(int)
        		maxima[maxima_null] = temp            
    
    # Filtre de Savitzky-Golay : moyenne glissante pondérée par un polynôme
	# pour mieux ajuster les valeurs au bord  
    
	maxima = sgn.medfilt(maxima, filter_order)  # filtre median pour enlever les pics
	top_row = sgn.savgol_filter(maxima, w_filter, sgolay_order)
	return top_row


def fit_curve2D(image, z_lim, w_filter, filter_order, sgolay_order):
	# Retourne les positions des max d'une image 2D : l'épithélium est la couche qui
	# renvoie le plus de lumière, rechercher les maxima en intensité sur l'image
	# permet donc de définir les contours de l'épithelium. Comme ces maxima varient
	# beaucoup d'une colonne à l'autre il est important d'utiliser une fonction de
	# lissage.
	colonne = np.shape(image)[1]
	maxima = np.zeros(colonne)
	for i in range(colonne):
		colonne_i = image[z_lim[0]:z_lim[1], i]
		maxima[i] = np.argmax(colonne_i[:])  # colonne_i.index(max(colonne_i[:]))

	# Filtre de Savitzky-Golay : moyenne glissante pondérée par un polynome
	# pour mieux ajuster les valeurs au bord

	maxima = sgn.medfilt(maxima, filter_order)  # filtre median pour enlever les pics
	top_row = sgn.savgol_filter(maxima, w_filter, sgolay_order)
#	plt.clf()
#	plt.pause(0.001)
	return top_row


def im2double(im):
	info = np.iinfo(im.dtype)  # Get the data type of the input image
	return im.astype(np.float) / info.max  # Divide all values by the largest possible value in the datatype
