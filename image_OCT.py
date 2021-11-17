import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as img
import scipy.signal as sgn
import os.path
from copy import deepcopy
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

from Fonction_fit import fit_curve2D_seuil, fit_curve2D

class image_OCT(object):
	def __init__(self, path):
		self.Path = path
		try:
			X = mpimg.imread(self.Path)
			extension = path.split('.')[-1]
			if extension == 'jpg':
				X = X[:, 2:, 0]*1.0             # float64 conversion (multiply by 1.0)
			elif extension == 'jpeg':
				X = X[:, 2:]*1.0

			if X.shape[1] == 1018:  # Test image size to define pixel size
				if np.mean(X[0:70, X.shape[1]-70:X.shape[1]]) - np.mean(X[0:40, 0:40]) > 4:
					target = 'Pachy'
				else:
					target = 'Line'
			elif X.shape[1] == 938:
				target='Corss'
			elif X.shape[1]==1534:
				target='PachyWide'
			else:
				message="\n"+"*"*50+"\n"+"Error in image_OCT init: File not found"
				raise ValueError(message)
			self.Path_eye = target
		except:
			message="\n"+"*"*50+"\n"+"Error in image_OCT init: File not found"
			raise ValueError(message)
		self.Path_patient = self.Path.split("\\")[-3]

	def ImageSNR(self,gauss_sigma=1):
		mImage=self.OCT
		signal = np.sum(np.square(mImage))
		noise = np.sum(np.square(mImage-img.gaussian_filter(mImage,sigma=gauss_sigma)))
		SNR = np.around(10*np.log10(signal/noise),1)
		return(SNR)
	
	def getImage(self,crop,z0=10):
		self.z0=z0 # [px] coordonnée axiale du démarrage de la détection (10)
		X = mpimg.imread(self.Path)
		self.OCT_brut = X[(self.z0):, (2+crop):(-crop), 0]*1.0             # conversion en float64 en multipliant par 1.0 ([z0:, 2:, 0]*1.0)
		if self.Path_eye == "PachyWide":  
			self.champ_acquisition_mm= 9 # full = 9mm pour PachyWide
			self.pas = 4.322                   # [µm] pas axial (!), cf. calibration via pachymetry Optovue
		elif self.Path_eye == "Pachy":  
			self.champ_acquisition_mm=6 # full = 6mm pour Pachy
			self.pas = 4.333                   # [µm] pas axial
		else:
			self.champ_acquisition_mm=8 # full = 8mm
			self.pas = 4.333                   # [µm] pas axial 

	def RemoveArrow(self,sz_arrow = 70):
		self.OCT_brut=self.OCT_brut
		self.OCT_brut[0:sz_arrow,np.shape(self.OCT_brut)[1]-sz_arrow:np.shape(self.OCT_brut)[1]] = np.zeros((sz_arrow,sz_arrow)) 

	def ExposureCorrection(self):
		self.OCT=self.OCT_brut
		hist_moy = np.mean(self.OCT_brut.ravel())
		if hist_moy > 25:
			hist_adjust = np.round(hist_moy - 18)
			self.OCT = self.OCT - hist_adjust
			self.OCT[self.OCT < 0] = 0

	def Derivative1(self,gauss_sigma=1):	
		OCT = img.gaussian_filter(self.OCT,sigma=gauss_sigma)
		self.mean_signal = np.mean(OCT, axis = 0)	
		self.mean_signal_smooth = sgn.savgol_filter(self.mean_signal,15,2)     # SG évite les discontinuités apparentes du filtre médian (qui feraient bugguer der1)
		self.der1 = np.diff(self.mean_signal_smooth)
		self.der1_smooth = sgn.savgol_filter(self.der1,15,2)

	def HyperRefelxionRemoval(self,gauss_sigma=1,der1_seuil=0.67,marge=55):
		pas_lat = round(1000*self.champ_acquisition_mm/self.OCT.shape[1],2)  # pas latéral
		marge = int(np.round(marge/pas_lat))
		self.Derivative1(gauss_sigma)
		offset_aff = np.ceil(np.amax(self.mean_signal)) + 5
		columns_raw = self.OCT.shape[1]
		coord_centre = np.where(self.mean_signal_smooth == np.amax(self.mean_signal_smooth[int(0.2*columns_raw):columns_raw-int(0.2*columns_raw)]))[0][0]

		if np.any(abs(self.der1[20:len(self.der1)-20]) > der1_seuil) :    # en évitant les bords            
				xmin_cut = np.where(self.der1_smooth == np.sort(self.der1_smooth[20:-20])[-1])[0][0] # (tuple to int)
				xmax_cut = np.where(self.der1_smooth == np.sort(self.der1_smooth[20:-20])[0])[0][0]
				self.xmin_cut = xmin_cut - marge
				self.xmax_cut = xmax_cut + marge    
				self.SpecularCut = (xmax_cut-xmin_cut)*pas_lat 
		else:    # pas de pic central détecté
				self.xmin_cut = 0
				self.xmax_cut = 0
				self.SpecularCut = 0             
		self.OCT_cut = np.delete(self.OCT, range(self.xmin_cut, self.xmax_cut), axis=1)
		self.depth, self.columns = np.shape(self.OCT_cut)

	def flattening(self,max_window_size=10,w_filter=101,median_filter_width=15,sgolay_order=2,w_filterNorm=115):
		SNR_2D=self.ImageSNR()
		seuil = int(-2*SNR_2D + 65) # (+ 65)            # [] seuil pour détection de l'épithélium (40)
		if self.Path_eye == "PachyWide":
		    seuil  = seuil - 10  #(-10)    
		elif np.any(self.der1_smooth[50:self.xmin_cut] > 0.25) or np.any(self.der1_smooth[self.xmax_cut:-50] > 0.25):
		    if SNR_2D < 16:
		        seuil  = seuil #- 5#(-5)    # OK le 08/02/21 sur Faustine2
		    else:                
		        seuil = seuil + 10 #(+ 15)
		else:
		    seuil = seuil + 15 #(+ 15)
		if np.any(self.OCT_cut[0:50,] > 1.5*seuil):
			if SNR_2D < 15.5:
				seuil  = seuil + 15                    
		else:
			seuil = seuil + 25
		# print('Epithelium detection threshold = %i'%seuil)
		OCT_cut_profile = img.gaussian_filter(self.OCT_cut, sigma=2)
		Displacement = abs(fit_curve2D_seuil(OCT_cut_profile, seuil, w_filter, median_filter_width, sgolay_order))
		# Correction de la coordonnée centrale de la cornée
		coord_centre2 = int(np.where(Displacement == np.amin(Displacement[100:self.columns-100]))[0][0])
		offset = int(round(min(Displacement)))  # décalage pour visualiser l'épithélium sur les images aplaties
		Displacement = Displacement - min(Displacement)
			# Application du décalage
		FlattenedImage = np.zeros((self.depth, self.columns))
		for j in range(self.columns):
			if int(round(Displacement[j])) > 0:
				FlattenedImage[:-int(round(Displacement[j])), j] = self.OCT_cut[int(round(Displacement[j])):, j]
			else:
				FlattenedImage[:, j] = self.OCT_cut[:, j]
			# Raffinement de l'aplatissement
		max_window = [0, max_window_size + offset]
		Displacement = abs(fit_curve2D(FlattenedImage, max_window, w_filter, median_filter_width, sgolay_order))   
		Displacement = Displacement - min(Displacement)
		FlattenedImage2 = np.zeros((self.depth, self.columns))
		for j in range(self.columns):
			if int(round(Displacement[j])) > 0:
				FlattenedImage2[:-int(round(Displacement[j])), j] = FlattenedImage[int(round(Displacement[j])):, j]
			else:
				FlattenedImage2[:, j] = FlattenedImage[:, j]
		sgolay_orderNorm = sgolay_order             # ordre du filtre de SG    
		M = FlattenedImage2[:max_window[1], :]
		VectMax = M.max(axis=0)
		VectNorm = sgn.savgol_filter(VectMax, w_filterNorm, sgolay_orderNorm)
		VectNorm[VectNorm<=75] = 75
		ProcessedImage = deepcopy(FlattenedImage2/VectNorm)
		self.OCT_flat=ProcessedImage

	def AutoTreatment(self,plot=False,crop=100):
		# if not plot:
		# 	plt.ioff()
		# else:
		# 	plt.ion()
		try:
			self.getImage(crop=crop)
			self.RemoveArrow()
			self.ExposureCorrection()
			self.HyperRefelxionRemoval()
			self.flattening()
		except:
			message="\n"+"*"*50+"\n"+"Error in image_OCT Autotreatment: Error during treatment"
			raise ValueError(message)

