 # -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:40:39 2020
Pre-processing algorithm for depth-resolved corneal images : flattens the image,
standardizes its illumination & plots the average intensity profile with depth.

Input = clinical OCT B-scan
Output = average OCT intensity profile with depth


@author: Maëlle
Oct. 13 2020: up-to-date version of pre-processing algo.
Nov. 17 2020: smoothed mask coordinates
Feb. 8 2021: figure refinement, output update (full stromal thickness), input folder GUI
, nouveau seuil pour dét. épith si SNR2D < 16 dB.
Feb. 11 2021: échelle bloquée sur fig.8 (avant/après correction)
Mar. 16 2021: limitation du champ visuel analysé (param. champ_lat ligne 85 => 22/09/21 champ_analyse_mm)
June 19 2021: ajout param. d'extension .jpg/.jpeg pour analyse simu Pierre (L.88) + L.546-547 coord_stroma
July 12 2021: ajout param. de largeur saturée sur signal de normalisation (épith)
Sept. 22 2021: modif lignes 598 et suivantes pour images décentrées (tq cadre ROI sort de l'image)
+ recalibration pachymetry : pas axial = 4.333 µm sauf pachywide 4.322 µm
"""
#%% 1/ INITIALISATION /////////////////////////////////////////////////////////
#%% 1.1 Environnement

# ATTENTION user : lines 126-128 (corr, save) et 108-110 (marge_postBowman, champ_analyse_mm)

#   reset
#del(specular_cut)

    # Libraries
#import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sgn
import scipy.ndimage as img
from matplotlib.widgets import Cursor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from detecta import detect_peaks
import tkinter as tk
from tkinter import filedialog
from copy import deepcopy
import csv
import time
# os.chdir("C:/Users/csoub/OneDrive/Bureau/3A/Ensta/Super projet/Algo_Maelle")
os.chdir("C:\\Users\\csoub\\OneDrive\\Bureau\\3A\\Ensta\\Super projet\\Algo_Maelle")
from Fonction_fit import fit_curve2D_seuil, fit_curve2D
from Fonction_auto import detect_stromal_peaks, cursor_segm_stroma, specular_cut_cursor
#from matplotlib.widgets import Cursor

plt.close('all')

    # Graphical parameters
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['figure.figsize'] = 14.5, 8  # figure size [inch]
fontsize_title = 20      # taille de la police des titres (plt)
fontsize_label = 15      # taille de la police des légendes d'axe (plt)
plt.rcParams.update({'font.size': fontsize_title})
plt.rcParams.update({'xtick.labelsize': 'xx-small'})
plt.rcParams.update({'ytick.labelsize': 'xx-small'})
plt.rcParams.update({'axes.labelsize': 'small'})

    # Clock
#t0_time = None
t0_time = time.time()

#%% 1.2 Variables (USER SET-UP)

# Default dictionary: values validated by use
dict = {}
dict.setdefault('z0', 10)
#dict.setdefault('seuil', 70) # (40 pour SNR haut, 70 pour SNR bas => cf 2.3)
dict.setdefault('max_window_size', 10)
dict.setdefault('profondeur_min', None)
dict.setdefault('profondeur_max', None)
dict.setdefault('median_filter_width',15)
dict.setdefault('w_filter',101)
dict.setdefault('sgolay_order',2)
dict.setdefault('w_filterNorm',115)
dict.setdefault('nLayers', 20) #(10)
dict.setdefault('w_filterLayers',275) # int ou None
#dict.setdefault('',)
# manquent der1_seuil, mph, mpd, threshold, marge_centre

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#-%%%%%%%%%%%%%%%%%%%%%%%%%%% To be filled by user %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

extension = '.jpg'

#Path_folder = 'C:\\Users\\Maëlle\\Documents\\Code - SD-OCT études\\_Point10 - bases de données\\comp9_database-normale-n85\\Faustine2\\'
#Path_patient = 'DRONIC_OS' # tag0 dans R
#Path_eye = 'Cross' # tag dans R
#Path_eye = 'Line' # tag dans R
#Path_eye = 'Pachy' # tag dans R
#Path_eye = 'PachyWide' # tag dans R
#Path_eye = '4'
# REMARQUE : en cas d'erreur 'IndexError: index 0 is out of bounds for axis 0 with size 0' => vérifier le chemin d'accès

    # Normalisation de l'image, segmentation du stroma, correction et sauvegarde
norm = 1                    # 0 : pas de normalisation latérale 
                            # 1 : normalisation par l'intensité lissée de l'épithélium 
                            # 2 : normalisation par l'intensité lissée de la 1ère sous-couche du stroma antérieur ==============> À IMPLÉMENTER/SUPPRIMER
#seg_stroma = 0              # 0 : segmentation manuelle du stroma
#                            # 1 : segmentation automatique du stroma
corr = True                 # correction de la focalisation imhomogène (True/False)
save = True                # sauvegarde images et fichier .txt (True/False)
show = False                 # affichage exhaustif des figures (si False, seulement les essentielles)

    # Gestion de l'hyperréflexion spéculaire
SpecularCut_option = 3      # 0 : valeurs des bornes spé'cifiées en entrée
                            # 1 : curseur de sélection manuelle des bornes de la zone spéculaire ==============================> À IMPLÉMENTER
                            # 2 : coupure de largeur fixe autour du centre de la cornée imagée
                            # 3 : détection automatique de la zone d'hyperréflexion spéculaire 
if SpecularCut_option == 3:
    der1_seuil = 0.67        # [] pente seuil de la dérivée pour la détection de l'artefact central (0.75)
    marge = 55               # [µm] marge de part et d'autre du pic central si détecté (10 px => 50 µm le 21/09/21 pour prendre en compte pas_lat)
#    marge_ratio = 1/4      # [] fraction de la zone centrale prise comme marge (1/7)
elif SpecularCut_option == 2:
    SpecularCut_um = 500    # [µm] Largeur fixée de la coupure spéculaire
elif SpecularCut_option == 0:
    xmin_cut = None            # première colonne que l'on veut enlever (réflexion spéculaire)  (None commenté)
    xmax_cut = None            # dernière colonne que l'on veut enlever  (None commenté)

    # Segmentation du stroma
marge_postBowman = 50          # [µm] marge de segmentation du stroma antérieur (50 µm)
marge_preDescemet = 30         # [µm] marge de segmentation endothélio-descemetique (Gatinel 12µm, ici 30 µm)
champ_analyse_mm = 6        # pour forcer l'analyse à X mm de champ transversal (à commenter si pas imposé)
mph = 0                        # [] minimum peak height [8bit]
mpd = 10                        # [px] minimum peak distance 
threshold = 0.01             # [] différence min avec voisins [0;1] (0.01)
    # Délimitation de l'artefact central
mph1 = None                    # [] minimum peak height [0;1]
mpd1 = 50                      # [px] minimum peak distance 
threshold1 = 0                 # [] différence min avec voisins [0;1]
marge_centre = 100             # [px] marge autour de la coord centrale pour délim. artefact (100)

    ### Valeurs par défaut (peuvent être modifiées ci-dessous par une nouvelle valeur)
z0 = dict['z0']                             # [px] coordonnée axiale du démarrage de la détection (10)
max_window_size = dict['max_window_size']   # [px] taille de la fenêtre pour trouver le max de l'épithélium (10)
profondeur_min = dict['profondeur_min']     # [px] segmentation inférieure du stroma sur la courbe d'intensité moyenne (None)
profondeur_max = dict['profondeur_max']     # [px] segmentation supérieure du stroma sur la courbe d'intensité moyenne (None)
#colonne_inf = None         # [px] sélection latérale inférieure de la zone d'intérêt (évite les 0 aux bords) (None)
#colonne_sup = None         # [px] sélection latérale supérieure de la zone d'intérêt (évite les 0 aux bords) (None)

# =============================================================================
#=============================== AUTOMATED ====================================
    # v5.1: GUI for folder
#root = tk.Tk()
#if 'Path_folder' not in locals():
##    pathname = filedialog.askdirectory
#    filepathname = filedialog.askopenfilename(initialdir="",  title='Sélectionnez l''image OCT que vous souhaitez analyser :')
#    dirname = filepathname.split("/")
#    Path_folder = '/'.join(dirname[:-3])
#else:
#    filepathname = filedialog.askopenfilename(initialdir=Path_folder,  title='Sélectionnez l''image OCT que vous souhaitez analyser :')
#    dirname = filepathname.split("/")
if 'Path_folder' not in locals():
    root = tk.Tk()
    Path_folder = filedialog.askdirectory(initialdir="\\",  title='Sélectionnez le dossier global de travail :', parent=root)
    filenames = glob.glob(Path_folder + '/**/*' + extension, recursive=True)
    fileindex = 0
    fileindex_tot = len(filenames)
    root.update()
    root.destroy()
dirname = filenames[fileindex].split('/')
if 'Main Report' in dirname[-1]:
    fileindex += 1
    dirname = filenames[fileindex].split('/')
dirname = dirname[:-1] + dirname[-1].split('\\')
pathname = '/'.join(dirname[:-1])
root = tk.Tk()
filepathname = filedialog.askopenfilename(initialdir=pathname, title='Sélectionnez l\'image OCT que vous souhaitez analyser :', parent=root)
root.update()
root.destroy()
Path_patient = filepathname.split('/')[-3]
Path_eye = filepathname.split('/')[-2]
if Path_patient != dirname[-3] or Path_eye != dirname[-2]:
    fileindex = np.where([Path_patient+'\\'+Path_eye in filename for filename in filenames])[0][0]    
print('Analyse en cours : '+Path_patient+'/'+Path_eye)
time_start = time.time()
pathname = Path_folder+'/'+Path_patient+'/'+Path_eye
    # Paramètre d'acquision de l'OCT
if Path_eye == "PachyWide":  
    dict.setdefault('champ_acquisition_mm', 9) # full = 9mm pour PachyWide
    pas = 4.322                   # [µm] pas axial (!), cf. calibration via pachymetry Optovue
elif Path_eye == "Pachy":  
    dict.setdefault('champ_acquisition_mm', 6) # full = 6mm pour Pachy
    pas = 4.333                   # [µm] pas axial
else:
    dict.setdefault('champ_acquisition_mm', 8) # full = 8mm
    pas = 4.333                   # [µm] pas axial 
if 'champ_analyse_mm' not in locals():
    champ_analyse_mm = dict['champ_acquisition_mm'] # si pas précisé, toute la largeur de l'image est analysée

    # Import OCT brut
#pathname = Path_folder + Path_patient + '\\' + Path_eye
#del dirname
filename = filepathname #glob.glob(pathname + '\\*' + extension)[-1]
X = mpimg.imread(filename)
if extension == '.jpg':
    OCT_brut = X[z0:, 2:, 0]*1.0             # conversion en float64 en multipliant par 1.0 ([z0:, 2:, 0]*1.0)
elif extension == '.jpeg':
    OCT_brut = X[z0:, 2:]*1.0
#x_start = 20                       # rogner l'image à gauche
#x_end = np.shape(X)[1]             # rogner l'image à droite
#OCT = OCT[:,x_start:x_end]

sz_arrow = 70
OCT_brut[0:sz_arrow,np.shape(OCT_brut)[1]-sz_arrow:np.shape(OCT_brut)[1]] = np.zeros((sz_arrow,sz_arrow)) # camoufler flèche acquisition
    
pas_lat = round(1000*dict['champ_acquisition_mm']/OCT_brut.shape[1],2)  # pas latéral
marge = int(np.round(marge/pas_lat))
coord_centre = None

fig_dpi = 150                       # export figures
gauss_sigma = 1                     # lissage gaussien 2D

if Path_eye == "Pachy":
    if OCT_brut.shape[1]/OCT_brut.shape[0] > 2:
        import sys
        sys.exit('Le fichier en cours d\'analyse est un PachyWide : changer le nom du dossier et redémarrez le noyau.\n'+pathname)
        

#%% 2/ PRÉ-TRAITEMENT /////////////////////////////////////////////////////////
#%% 2.0 SNR 2D
signal = np.sum(np.square(OCT_brut))
noise = np.sum(np.square(OCT_brut-img.gaussian_filter(OCT_brut, sigma=gauss_sigma)))
SNR_2D = np.around(10*np.log10(signal/noise),1)
#print(gauss_sigma)
print('SNR_2D = %.1f dB'%SNR_2D)

## SNR with depth ?
#from scipy.stats import pearsonr, spearmanr
#SNR_z = []
#for i in range(ROI.shape[0]):
#    signal_z = np.sum(np.square(ROI[i,:]))
#    noise_z = np.sum(np.square(ROI[i,:]-img.gaussian_filter1d(ROI[i,:], sigma=2)))
#    SNR_z.append(np.around(10*np.log10(signal_z/noise_z),1))
#z = np.arange(0,ROI.shape[0]*pas,pas)
#z_i = 100
#plt.plot(ROI[z_i,:])
#plt.plot(img.gaussian_filter1d(ROI[z_i,:], sigma=2))
#plt.figure()
#plt.plot(z,SNR_z)
#spearmanr(z,SNR_z) 
## test corrélation Pearson z et SNR_z => p-value PAS significative...

#%% 2.1 Correction de l'exposition

hist_moy = np.mean(OCT_brut.ravel())
if hist_moy > 25:
    hist_adjust = np.round(hist_moy - 18)
    OCT_optim = OCT_brut - hist_adjust
    OCT_optim[OCT_optim < 0] = 0
    if show == True:
        fig, axs = plt.subplots(2,2)
        fig.suptitle('Optimisation de l\'image OCT via l\'histogramme')
        axs[0,0].imshow(OCT_brut, cmap="gray")
        axs[0,0].axis('off')
        axs[1,0].hist(OCT_brut.ravel(),256,[0,256], density=True) 
        axs[1,0].set(xlabel = 'Niveau de gris 8-bits', ylabel = 'Densité')
        axs[0,1].imshow(OCT_optim, cmap = 'gray')
        axs[0,1].axis('off')
        axs[1,1].hist(OCT_optim.ravel(),256,[0,256], density=True)
        axs[1,1].set(xlabel = 'Niveau de gris 8-bits')  
        plt.show()
    OCT_brut = OCT_optim

# Suppl : affichage de l'axe y en %
#import matplotlib.ticker as mptick 
#from matplotlib.ticker import PercentFormatter 
#bin_counts, bin_edges = np.histogram(OCT.ravel(),256,[0,256])
#fig = plt.hist(bin_counts, bins=256, range=[0, 255], density=True)
#yticks = mptick.PercentFormatter(1.0, decimals=0, is_latex=True) 
#axs[1,0].yaxis.set_major_formatter(yticks)
    
    signal = np.sum(np.square(OCT_optim))
    noise = np.sum(np.square(OCT_brut-img.gaussian_filter(OCT_optim, sigma=gauss_sigma)))
    SNR_2D_optim = np.around(10*np.log10(signal/noise),1)

#%% 2.2 Gestion de l'hyperréflexion spéculaire
OCT = img.gaussian_filter(OCT_brut, sigma=gauss_sigma)
mean_signal = np.mean(OCT, axis = 0)

###### Coupure de LARGEUR FIXE ################################################
if SpecularCut_option == 2:
    # Coordonnée du centre de la cornée
    mean_signal_smooth = sgn.savgol_filter(mean_signal,115,2)     # SG évite les discontinuités apparentes du filtre médian (qui feraient bugguer der1)
    der1 = np.diff(mean_signal_smooth)
    der1_smooth = sgn.savgol_filter(der1,35,2)
    der1_annulneg = np.where(np.diff(np.sign(der1_smooth))<0)[0]
    coord_centre = der1_annulneg[np.argmin(abs(der1_annulneg - np.where(mean_signal_smooth == max(mean_signal_smooth))[0][0]))] #OCT.shape[1]/2))]

    SpecularCut = SpecularCut_um/pas_lat
    xmin_cut = int(round(coord_centre - SpecularCut/2))
    xmax_cut = int(round(coord_centre + SpecularCut/2))
    # Affichage
    fig = plt.figure(2)
    plt.imshow(OCT_brut,cmap = 'gray')
    offset_aff = np.ceil(np.amax(mean_signal)) + 5
    
    coeff = 30/(np.amax(der1)-np.amin(der1))
    lineM, = plt.plot(offset_aff -mean_signal_smooth, color='tab:purple')
    lineN, = plt.plot(offset_aff -coeff*der1_smooth, color='tab:orange')
    plt.axhline(offset_aff, xmin=0, xmax=OCT.shape[1]-1, color='tab:orange', linewidth=0.3)
    lineO, = plt.plot(np.ones([len(OCT),1]), color='tab:orange', linestyle = 'dashed', linewidth=1.5)
    lineP, = plt.plot(np.ones([len(OCT),1]), color='r', linestyle = 'dashed', linewidth=1)        
    plt.axvline(xmin_cut, ymin=0, ymax=OCT.shape[0]-1, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(xmax_cut, ymin=0, ymax=OCT.shape[0]-1, color='r', linestyle='dashed', linewidth=1)
    plt.title('Coupure spéculaire fixe : largeur = %i µm' %SpecularCut_um)
    plt.axvline(coord_centre, ymin=0, ymax=OCT.shape[1]-1, color='tab:orange', linestyle = 'dashed', linewidth = 1.5)
    plt.legend((lineM, lineN, lineO, lineP),('Signal latéral moyen lissé',
                   'Dérivée du signal latéral moyen lissé', 'Centre de la cornée imagée : x = %i' %coord_centre, 
                   'Coupure spéculaire de largeur fixe'), fontsize=11, loc = 'lower right')   
    lineO.set_visible(False)
    lineP.set_visible(False) 
    if save == True:
        plt.savefig(pathname + '\\1_SpecularCut_fixe.png', dpi=fig_dpi, bbox_inches='tight')

###### Coupure de LARGEUR AUTOMATIQUE #########################################
elif SpecularCut_option == 3:    
    mean_signal_smooth = sgn.savgol_filter(mean_signal,15,2)     # SG évite les discontinuités apparentes du filtre médian (qui feraient bugguer der1)
    der1 = np.diff(mean_signal_smooth)
    der1_smooth = sgn.savgol_filter(der1,15,2)
    #der2 = np.diff(der1_smooth)
    
    fig = plt.figure(2)
    plt.imshow(OCT_brut,cmap = 'gray')
    offset_aff = np.ceil(np.amax(mean_signal)) + 5
    #coeff = 20/(np.amax(der1)-np.amin(der1))
    lineN, = plt.plot(offset_aff + -mean_signal_smooth, color='tab:purple')
    lineO, = plt.plot(offset_aff + -10*der1, color='tab:green')
    columns_raw = OCT_brut.shape[1]
    coord_centre = np.where(mean_signal_smooth == np.amax(mean_signal_smooth[int(0.2*columns_raw):columns_raw-int(0.2*columns_raw)]))[0][0]
    
    if np.any(abs(der1[20:len(der1)-20]) > der1_seuil) :    # en évitant les bords            
        xmin_cut = np.where(der1_smooth == np.sort(der1_smooth[20:-20])[-1])[0][0] # (tuple to int)
        xmax_cut = np.where(der1_smooth == np.sort(der1_smooth[20:-20])[0])[0][0]
#        marge = int(np.floor((xmax_cut - xmin_cut)*marge_ratio))     # marge = 1/Xe de la largeur du pic central
        xmin_cut = xmin_cut - marge
        xmax_cut = xmax_cut + marge    
        plt.axvline(xmin_cut, ymin=0, ymax=OCT.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=1)
        plt.axvline(xmax_cut, ymin=0, ymax=OCT.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=1)
        lineP, = plt.plot(np.ones([len(OCT),1]), color='tab:green', linestyle = 'dashed', linewidth=1)
        SpecularCut = (xmax_cut-xmin_cut)*pas_lat
        plt.title("Coupure spéculaire auto : largeur = %i µm" %SpecularCut)
        plt.legend((lineN, lineO, lineP),('Signal latéral moyen lissé',
                   'Dérivée du signal latéral moyen lissé', 'Délimitation de l\'artefact spéculaire'), fontsize=11, loc = 'lower right')   
        lineP.set_visible(False) 
    else:    # pas de pic central détecté
        xmin_cut = 0
        xmax_cut = 0
        SpecularCut = 0              
#        lineP, = plt.plot(np.ones([len(OCT),1]), color='tab:green', linestyle = 'dashed', linewidth=1.5)
#        plt.axvline(coord_centre, ymin=0, ymax=OCT.shape[1]-1, color='tab:green', linestyle = 'dashed', linewidth = 1.5)
        plt.title('Détection auto de l\'artefact central : largeur = 0 µm')
#        plt.legend((lineN, lineO, lineP),('Signal latéral moyen lissé', 'Dérivée du signal latéral moyen lissé', 'Centre de la cornée imagée : x = %i' %coord_centre), fontsize=11, loc = 'lower right') 
        plt.legend((lineN, lineO),('Signal latéral moyen lissé', 'Dérivée du signal latéral moyen lissé'), fontsize=11, loc = 'lower right') 
#        lineP.set_visible(False)
#    if save == True:
#        plt.savefig(pathname + '\\1_SpecularCut_auto_%ium.png'%((xmax_cut-xmin_cut)*pas_lat), dpi=fig_dpi, bbox_inches='tight')
#    print("SpecularCut = %i µm" %((xmax_cut-xmin_cut)*pas_lat))

    
###### Coupure déterminée au CURSEUR MANUEL ###################################
elif SpecularCut_option == 1:
    fig = plt.figure(2)
    ax = fig.add_subplot(111, facecolor='w')
    plt.imshow(OCT_brut, cmap="gray")
    plt.title('Sélectionnez les bornes latérales du pic spéculaire')
    fig.cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='g', linewidth=1)
    R = plt.ginput(2)
    xmin_cut = int(min(round(R[0][0]), round(R[1][0])))
    xmax_cut = int(max(round(R[0][0]), round(R[1][0])))
    SpecularCut = int(round((xmax_cut - xmin_cut)*pas))
    coord_centre = xmin_cut + np.where(mean_signal[xmin_cut:xmax_cut] == np.amax(mean_signal[xmin_cut:xmax_cut]))[0][0]
#    OCT_cut = np.delete(OCT, range(xmin_cut, xmax_cut), axis=1)
    plt.clf()
    plt.imshow(OCT_brut, cmap="gray")
    plt.axvline(xmin_cut, ymin=0, ymax=OCT.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=1)
    plt.axvline(xmax_cut, ymin=0, ymax=OCT.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=1)
    lineQ, = plt.plot(np.ones([len(OCT),1]), color='tab:green', linestyle = 'dashed', linewidth=1, label='Délimitation de l\'artefact spéculaire')
    plt.legend(fontsize=11, loc = 'lower right')   
    lineP.set_visible(False)
    plt.title("Coupure spéculaire manuelle : largeur = %i µm" %SpecularCut) 
       
plt.show()
    
#%%  2.2bis Validation de la coupure centrale

time_spec_end = time.time()    
#import tkinter as tk
#from tkinter import messagebox
result = None
temp = None
temp_exit = None
root = tk.Tk()
canvas2 = tk.Canvas(root, width = 300, height = 0)
#canvas1.place(x=10000, y=15000)
canvas2.pack()

def CutOK():
    # accepter ici nouvelles valeurs de segmentation ?    
    root.destroy()
def CutNotOK():
    MsgBox = tk.messagebox.askokcancel('Coupure refusée', 'Coupure centrale : Oui=à la main / Annuler=pas de coupure', icon = 'question')
    if MsgBox:
        global result
        plt.figure(2)
        plt.close()
        result = specular_cut_cursor(OCT_brut, mean_signal, mean_signal_smooth, der1, pathname, pas_lat)
        return result
#        print(result)
    else:
        plt.figure(2)
        plt.clf()
        plt.imshow(OCT_brut,cmap = 'gray')
        offset_aff = np.ceil(np.amax(mean_signal)) + 5
        #coeff = 20/(np.amax(der1)-np.amin(der1))
        lineN, = plt.plot(offset_aff + -mean_signal_smooth, color='tab:purple')
        lineO, = plt.plot(offset_aff + -10*der1, color='tab:green')
        plt.title('Détection auto de l\'artefact central : largeur = 0 µm')
#        plt.legend((lineN, lineO, lineP),('Signal latéral moyen lissé', 'Dérivée du signal latéral moyen lissé', 'Centre de la cornée imagée : x = %i' %coord_centre), fontsize=11, loc = 'lower right') 
        plt.legend((lineN, lineO),('Signal latéral moyen lissé', 'Dérivée du signal latéral moyen lissé'), fontsize=11, loc = 'lower right') 
        global temp#SpecularCut, xmin_cut1, xmax_cut1
        temp = 1
#        SpecularCut_option = 1
#        SpecularCut = 0
#        xmin_cut1 = 0
#        xmax_cut1 = 0
        root.destroy()
        return temp #SpecularCut, SpecularCut_option, xmin_cut1, xmax_cut1
#    root.lift()
#        exit()
def ExitApp():
#    global temp_exit
#    temp_exit = 1
    root.destroy()
    raise SystemExit(...)
#    return temp_exit
#    exit()

#if temp_exit != None:
#    raise SystemExit(...)

button1 = tk.Button(root, text='Coupure acceptée',command=CutOK,bg='green',fg='white').pack(fill=tk.X)
canvas2.create_window(150, 50, window=button1)
button2 = tk.Button(root, text='Coupure refusée',command=CutNotOK,bg='brown',fg='white').pack(fill=tk.X)
canvas2.create_window(150, 100, window=button2)
button3 = tk.Button(root, text='Interrompre le code',command=ExitApp,bg='red',fg='white').pack(fill=tk.X)
canvas2.create_window(150, 150, window=button3)
  
root.mainloop()

coord_centre_img = 0
if result != None:
    xmin_cut = result[0]
    xmax_cut = result[1]
    coord_centre = result[2]
    SpecularCut = result[3]       
    coord_centre_img = 0
    SpecularCut_option = 1

if temp != None:
    SpecularCut_option = 1
    SpecularCut = 0
    xmin_cut = 0
    xmax_cut = 0 
        
if save == True:
    if SpecularCut_option == 3:
        fig = plt.figure(2)
        plt.savefig(pathname + '\\1_SpecularCut_auto_%ium.png'%(np.round((xmax_cut-xmin_cut)*pas_lat)), dpi=fig_dpi, bbox_inches='tight')
    elif SpecularCut_option == 1:
        fig = plt.figure(2)
        plt.savefig(pathname + '\\1_SpecularCut_manuelle_%ium.png'%(np.round((xmax_cut-xmin_cut)*pas_lat)), dpi=fig_dpi, bbox_inches='tight')
    
#==============================================================================        
    # Découpage de l'image OCT
if 'xmin_cut' in locals(): # ce qui devrait toujours être le cas
    OCT_cut_brut = np.delete(OCT_brut, range(xmin_cut, xmax_cut), axis=1)
#    plt.clf()
#    plt.figure(3)
#    plt.imshow(OCT_cut, cmap="gray")
    depth, columns = np.shape(OCT_cut_brut)

#OCT = OCT_cut

#%% 2.3 Aplatissement de la cornée
time_flat_start = time.time() 
seuil = int(-2*SNR_2D + 65) # (+ 65)            # [] seuil pour détection de l'épithélium (40)
if SpecularCut != 0:
    if Path_eye == "PachyWide":
        seuil  = seuil - 10  #(-10)    
    elif np.any(der1_smooth[50:xmin_cut] > 0.25) or np.any(der1_smooth[xmax_cut:-50] > 0.25):
        if SNR_2D < 16:
            seuil  = seuil #- 5#(-5)    # OK le 08/02/21 sur Faustine2
        else:                
            seuil = seuil + 10 #(+ 15)
    else:
        seuil = seuil + 15 #(+ 15)
    
else:
    if np.any(OCT[0:50,] > 1.5*seuil):
        if SNR_2D < 15.5:
            seuil  = seuil + 15                    
        else:
            seuil = seuil + 25
#seuil = 30
print('seuil dét. épithélium = %i'%seuil)

OCT_cut = img.gaussian_filter(OCT_cut_brut, sigma=2)

    # Détection de l'épithelium
w_filter = dict['w_filter']                         # largeur du filtre Savitzky-Golay : doit être impair
median_filter_width = dict['median_filter_width']   # taille de la fenêtre glissante pour filtre médian (5)
sgolay_order = dict['sgolay_order']                 # ordre du filtre de Savitzky-Golay (< fenêtre)

fig = plt.figure(3)
plt.imshow(OCT_cut, cmap = 'gray')
plt.title('Détection de l\'épithélium')

Displacement = abs(fit_curve2D_seuil(OCT_cut, seuil, w_filter, median_filter_width, sgolay_order))
# Correction de la coordonnée centrale de la cornée
coord_centre2 = int(np.where(Displacement == np.amin(Displacement[100:columns-100]))[0][0])
offset = int(round(min(Displacement)))  # décalage pour visualiser l'épithélium sur les images aplaties
Displacement = Displacement - min(Displacement)

    # Application du décalage
FlattenedImage = np.zeros((depth, columns))
for j in range(columns):
	if int(round(Displacement[j])) > 0:
		FlattenedImage[:-int(round(Displacement[j])), j] = OCT_cut_brut[int(round(Displacement[j])):, j]
	else:
		FlattenedImage[:, j] = OCT_cut_brut[:, j]
    # Raffinement de l'aplatissement
max_window = [0, max_window_size + offset]
Displacement = abs(fit_curve2D(FlattenedImage, max_window, w_filter, median_filter_width, sgolay_order))   
plt.legend(('Détection des maxima locaux', 'Signal de l\'épithélium lissé', 'Offset brut', 'Offset lissé'), fontsize=11, loc = 'lower right')
Displacement = Displacement - min(Displacement)
FlattenedImage2 = np.zeros((depth, columns))
for j in range(columns):
	if int(round(Displacement[j])) > 0:
		FlattenedImage2[:-int(round(Displacement[j])), j] = FlattenedImage[int(round(Displacement[j])):, j]
	else:
		FlattenedImage2[:, j] = FlattenedImage[:, j]
        
##%% 2.5 Segmentation du stroma
seg_stroma = 1
OCT = OCT_cut_brut[:]
#ProcessedImage = deepcopy(FlattenedImage2)
Mean_I_large = np.mean(FlattenedImage2, axis=1)

    # Segmentation axiale par détection de pics
#if profondeur_min is None:
coord_stroma, coord_cornee = detect_stromal_peaks(Mean_I_large, mph, mpd, threshold, edge = 'both', aff_peaks = show)
#coord_stroma = (25,139) # Simu Pierre
ep_cornee_um = int(abs(coord_cornee[0]*pas-coord_cornee[1]*pas))
ep_stroma_tot_um = int(abs(coord_stroma[1]*pas-coord_stroma[0]*pas))
#    if coord_stroma[0] < offset:   # erreur si le pic de Bowman est avant l'épithélium
#        raise ValueError('Erreur de détection des bornes du stroma : pics détectés avant l\'épithélium')
    # ==========================================================================================>>>>>>>>>> proposer sélection manuelle sur image OCT <<<<<<<<<<<<===
profondeur_min_um = coord_stroma[0]*pas + marge_postBowman
profondeur_max_um = coord_stroma[1]*pas - marge_preDescemet
ep_stroma_analyse_um = int(abs(round(profondeur_max_um-profondeur_min_um)))

profondeur_min = int(round(profondeur_min_um/pas))
profondeur_max = int(round(profondeur_max_um/pas))

    # Segmentation latérale via absence de signal sur dernière ligne
idx_nul_min = np.where(sgn.medfilt(img.gaussian_filter(FlattenedImage2, sigma=0.5)[profondeur_min,],5)==0)   # filtre médian pour éviter détection du bruit
idx_nul_max = np.where(sgn.medfilt(img.gaussian_filter(FlattenedImage2, sigma=0.5)[profondeur_max,],5)==0)   # filtre médian pour éviter détection du bruit
colonne_inf_min = idx_nul_min[0][idx_nul_min[0] < int(columns/2)]  # étape intermédiaire pour éviter pb de segmentation si aplatissement raté au bord
colonne_inf_max = idx_nul_max[0][idx_nul_max[0] < int(columns/2)]
colonne_sup_min = idx_nul_min[0][idx_nul_min[0] > int(columns/2)]  
colonne_sup_max = idx_nul_max[0][idx_nul_max[0] > int(columns/2)]
if len(colonne_inf_max) != 0 and len(colonne_inf_min) != 0:
    colonne_inf = np.amax([np.amax(colonne_inf_min),np.amax(colonne_inf_max)]) +2  # (indexation via tuple)
elif len(colonne_inf_max) != 0:
    colonne_inf = np.amax(colonne_inf_max) +2
else:
    colonne_inf = 0
        
if len(colonne_sup_max) != 0 and len(colonne_sup_min) != 0:
    colonne_sup = np.amin([np.amin(colonne_sup_min),np.amin(colonne_sup_max)]) -2  # (indexation via tuple)
elif len(colonne_sup_max) != 0:
    colonne_sup = np.amin(colonne_sup_max) -2
else:
    colonne_sup = columns
    
    # Si analyse sur diamètre restreint (champ visuel réduit)
if champ_analyse_mm < dict['champ_acquisition_mm']:
    rayon = int(np.round(0.5*columns_raw*champ_analyse_mm/dict['champ_acquisition_mm']))
    rayon_cut = SpecularCut/(2*pas_lat)
    if coord_centre2-rayon > colonne_inf and colonne_sup-coord_centre2 > rayon:
        colonne_inf = int(coord_centre2 - rayon + rayon_cut)
        colonne_sup = int(coord_centre2 + rayon - rayon_cut)
    else:
        if coord_centre2-rayon < colonne_inf:
            colonne_sup = int(colonne_inf + 2*rayon - 2*rayon_cut)
        elif colonne_sup-coord_centre2 < rayon:
            colonne_inf = int(colonne_sup - 2*rayon + 2*rayon_cut)
    
    # Affichage ###############################################################
CutImage = deepcopy(FlattenedImage2[:profondeur_max+100, colonne_inf:colonne_sup])
Mean_I_large_adjust = np.mean(CutImage, axis=1)
SD_I_large_adjust = np.std(CutImage, axis=1)#/np.sqrt(np.shape(CutImage)[1]-1)

#% Tracé de la fonction profil avec segmentation
N_pix_depth = np.shape(Mean_I_large_adjust)[0]
z = np.arange(0, N_pix_depth*pas, pas)[:N_pix_depth]

fig = plt.figure(5)
fig.clf()
ax = fig.add_subplot(211, facecolor='w')
ax.plot(z, Mean_I_large_adjust, linewidth=2)
#ax.errorbar(z, Mean_I_large_adjust,SD_I_large_adjust)
plt.fill_between(z, Mean_I_large_adjust-SD_I_large_adjust, Mean_I_large_adjust+SD_I_large_adjust,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
    linewidth=0)
segm_x = (coord_cornee[0],coord_stroma[0],coord_stroma[1])
ax.plot(tuple(pas*np.array(segm_x)), Mean_I_large_adjust[(segm_x),], '+', mfc=None, mec='r', mew=2, ms=8)
# attention virgule indispensable après tuple pour l'indexation de Mean_I_large
plt.axvline(profondeur_min_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
plt.axvline(profondeur_max_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
plt.title('Segmentation du stroma en profondeur')
text_str = 'Épaisseur totale cornée = %i µm\n Épaisseur totale stroma = %i µm (%.1f%%)\n Épaisseur stroma analysée = %i µm (%.1f%%)\n Champ analysé = %.1f mm' %(ep_cornee_um,ep_stroma_tot_um,100*round(ep_stroma_tot_um/ep_cornee_um,3),ep_stroma_analyse_um,100*round(ep_stroma_analyse_um/ep_cornee_um,3),np.round(((colonne_sup-colonne_inf)*pas_lat+SpecularCut)/1000,1))
(ymin,ymax) = ax.get_ylim()
plt.text(np.amax(z)*1.04,0.7*ymax,text_str,horizontalalignment='right',fontsize=9)
plt.xlabel('Profondeur [µm]', fontsize=15)
plt.ylabel('Intensité OCT moyenne', fontsize=15)
    
ax = fig.add_subplot(212)
plt.imshow(FlattenedImage2[:profondeur_max+100,], cmap="gray")
plt.hlines(profondeur_min, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
plt.hlines(profondeur_max, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
plt.vlines(colonne_inf, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
plt.vlines(colonne_sup, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
plt.axis('off')
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
if save == True:
    plt.savefig(pathname + '/2_segmentation_stroma_champ_%.1fmm_std.png'%champ_analyse_mm, dpi=fig_dpi, bbox_inches='tight')
    
    # ROI sur profil
#if save == True:
    fig = plt.figure(6)
    fig.clf()
    ax = fig.add_subplot(111, facecolor='w')
    ax.plot(z, Mean_I_large_adjust, linewidth=2)
    ax.plot(tuple(pas*np.array(coord_stroma)),Mean_I_large_adjust[(coord_stroma),], '+', mfc=None, mec='r', mew=2, ms=8)
    # attention virgule indispensable après tuple pour l'indexation de Mean_I_large
    plt.axvline(profondeur_min_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(profondeur_max_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
    plt.title('Segmentation axiale du stroma')
    plt.xlabel('Profondeur [µm]', fontsize=fontsize_label)
    plt.ylabel('Intensité OCT moyenne normalisée', fontsize=fontsize_label)
    plt.savefig(pathname + '/2_segmentation_stroma_profil.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # ROI sur profil à la verticale !
#    fig = plt.figure()    
#    ax = fig.add_subplot(111, facecolor='w')
#    ax.plot(Mean_I_large_adjust, z, linewidth=2)
#    ax.plot(Mean_I_large_adjust[(segm_x),],tuple(pas*np.array(segm_x)), '+', mfc=None, mec='r', mew=2, ms=8, label='Corneal segmentation')
#    plt.ylim((500,1750))
#    plt.gca().invert_yaxis()
#    plt.axhline(profondeur_min_um, xmin=0, xmax=1, color='r', linestyle='dashed', linewidth=1, label='Stromal ROI')
#    plt.axhline(profondeur_max_um, xmin=0, xmax=1, color='r', linestyle='dashed', linewidth=1)
##    plt.title('In depth SD-OCT mean intensity')
#    plt.ylabel('Depth [µm]', fontsize=fontsize_label)
#    plt.xlabel('Average coherently backscattered intensity [a.u.]', fontsize=fontsize_label)
#    plt.legend(fontsize=13)
    
    
    # ROI sur image OCT
#if save == True:
#    fig = plt.figure(7)
#    plt.clf()
#    plt.imshow(FlattenedImage2, cmap="gray")
#    ax = fig.add_subplot(1, 1, 1)
#    plt.hlines(profondeur_min, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
#    plt.hlines(profondeur_max, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
#    plt.vlines(colonne_inf, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
#    plt.vlines(colonne_sup, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
#    plt.axis('off')
#    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#    plt.savefig(pathname + '/2_segmentation_stroma_ROI.png', dpi=100, bbox_inches=extent)
#    plt.close()
    
plt.show()

#%%  2.5bis Validation de la segmentation
time_flat_end = time.time() 
#import tkinter as tk
#from tkinter import messagebox
result = None
temp_exit = None
root = tk.Tk()
canvas1 = tk.Canvas(root, width = 300, height = 0)
#canvas1.place(x=10000, y=15000)
canvas1.pack()

def SegOK():
    # accepter ici nouvelles valeurs de segmentation ?
    root.destroy()
def SegNotOK():
    global result    
    MsgBox = tk.messagebox.askquestion('Passage en mode manuel', 'Segmentation :  Oui=sur le profil / Non=sur sur l\'image OCT', icon = 'question')
    fig = plt.figure(5)
    plt.close()
#    fig = plt.figure(6)
#    plt.close()
#    fig = plt.figure(7)
#    plt.close() 
    if MsgBox:
        result = cursor_segm_stroma(MsgBox, FlattenedImage2, marge_postBowman, marge_preDescemet, SpecularCut, colonne_inf, colonne_sup, coord_cornee, ep_stroma_tot_um, ep_stroma_analyse_um, pas, pas_lat, save, pathname, champ_analyse_mm, coord_centre2)        
        return result
        print(result)
#    root.lift()
#        canvas1.show() ou truc du genre
#    else:
#        root.destroy()
#        exit()
def ExitApp():
#    global temp_exit
#    temp_exit = 1
    root.destroy()
    raise SystemExit(...)
#    exit()
#    return temp_exit

#if temp_exit != None:
#    raise SystemExit(...)

button1 = tk.Button(root, text='Segmentation acceptée',command=SegOK,bg='green',fg='white').pack(fill=tk.X)
canvas1.create_window(150, 50, window=button1)
button2 = tk.Button(root, text='Passage en mode manuel',command=SegNotOK,bg='brown',fg='white').pack(fill=tk.X)
canvas1.create_window(150, 100, window=button2)
button3 = tk.Button(root, text='Interrompre le code',command=ExitApp,bg='red',fg='white').pack(fill=tk.X)
canvas1.create_window(150, 150, window=button3)
  
root.mainloop()

if result != None:
    colonne_inf = result[0]
    colonne_sup = result[1]
    coord_stroma = result[2]
    coord_cornee = (coord_cornee[0],coord_stroma[1])
    profondeur_min = result[3]
    profondeur_max = result[4]
    profondeur_min_um = result[5]
    profondeur_max_um = result[6]
    seg_stroma = 0
    
    ep_cornee_um = int(abs(coord_cornee[0]*pas-coord_cornee[1]*pas))
    ep_stroma_tot_um = int(abs(coord_stroma[1]*pas-coord_stroma[0]*pas))
    ep_stroma_analyse_um = int(profondeur_max_um-profondeur_min_um)
    
del(result)

print('épaisseur totale de cornée = %i µm\népaisseur relative du stroma = %i%%\népaisseur relative analysée = %i%%' %(ep_cornee_um, 100*round(ep_stroma_tot_um/ep_cornee_um,3), 100*round(ep_stroma_analyse_um/ep_cornee_um,3)))
     
#%% 3/ CORRECTION A POSTERIORI ////////////////////////////////////////////////
#%%  3.1 Découpage du stroma en strates successives
time_corr_start = time.time() 

profondeur_Bowman = coord_stroma[0]
profondeur_endoth = coord_stroma[1]

if corr == True:    
        # Coordonnées des strates
    N_layers = dict['nLayers']
#    N_layers = 15
    colors = [plt.cm.jet(i) for i in np.linspace(0, 1, N_layers+1)]   # gradient de couleurs !
    depth_stroma_large = int(round((profondeur_endoth - profondeur_Bowman)*pas))
    coord_layers = np.around(np.linspace(profondeur_Bowman, profondeur_endoth, N_layers+1))
    coord_layers = coord_layers.astype(int)
        # Calcul du signal moyen pour chaque strate
    Mean_I_layers = np.zeros((len(coord_layers)-1,colonne_sup-colonne_inf))
    w_filterLayers = dict['w_filterLayers']
    for i in range(0,len(coord_layers)-1):
        if w_filterLayers != None:      # Lissage
            Mean_I_layers[i,:] = sgn.savgol_filter(np.mean(FlattenedImage2[coord_layers[i]:coord_layers[i+1],colonne_inf:colonne_sup], axis=0), w_filterLayers, 2)
        else:                           # Pas de lissage
            Mean_I_layers[i,:] = np.mean(FlattenedImage2[coord_layers[i]:coord_layers[i+1],colonne_inf:colonne_sup], axis=0)
            
        # Affichage
    if show == True and save == False:
        plt.figure(8)
        ax = plt.axes()
        ax.set_prop_cycle('color', colors)
        lineM = plt.plot(Mean_I_layers.T, linewidth = 1)
        plt.legend(lineM, range(1,len(coord_layers)), title='n° layer', fontsize = 'xx-small', loc='center left', bbox_to_anchor=(0.95, 0.5))
#        plt.title('Evolution du signal lissé non normalisé en profondeur')
#        plt.xlabel('Coordonnée latérale [px]', fontsize=fontsize_label)
#        plt.ylabel('Intensité OCT moyenne brute', fontsize=fontsize_label)
        plt.title('Smoothed signal of N=%i in-depth stromal layers' %N_layers)
        plt.xlabel('Lateral coordinate [px]', fontsize=fontsize_label)
        plt.ylabel('Raw mean OCT intensity', fontsize=fontsize_label)
        plt.show()
    
#    if save == True:
#        plt.figure(8)
#        ax = plt.axes()
#        ax.set_prop_cycle('color', colors)
#        lineM = plt.plot(Mean_I_layers.T, linewidth = 1)
#        plt.legend(lineM, range(1,len(coord_layers)), title='N layer', fontsize = 'x-small', loc='center left', bbox_to_anchor=(0.95, 0.5))
#        plt.title('Evolution du signal lissé non normalisé en profondeur')
#        plt.xlabel('Coordonnée latérale [px]', fontsize=fontsize_label)
#        plt.ylabel('Intensité OCT moyenne brute', fontsize=fontsize_label)
#        plt.savefig(pathname + '\\4_corr_%i_layers_avant.png' %N_layers)
#        plt.close()
    
        # Affichage découpage OCT
    if show == True:
        fig = plt.figure(9)
        ax = fig.add_subplot(111, facecolor='#FFFFCC')
        plt.imshow(FlattenedImage2, cmap="gray")
#        plt.title('Découpage du stroma')
        plt.title('Stromal in-depth slicing')
        plt.axis('off')
        
        plt.figure(9)
        ln_width = 0.45
        for i in range(1,len(coord_layers)-2):
            plt.hlines(coord_layers[i], xmin=colonne_inf, xmax=colonne_sup, color=colors[i], linestyle='dashed', linewidth=ln_width, label='_nolegend_')
            plt.hlines(coord_layers[i+1], xmin=colonne_inf, xmax=colonne_sup, color=colors[i], linestyle='dashed', linewidth=ln_width, label='_nolegend_')
            plt.vlines(colonne_inf, ymin=coord_layers[i], ymax=coord_layers[i+1], color=colors[i], linestyle='dashed', linewidth=ln_width, label='_nolegend_')
            plt.vlines(colonne_sup, ymin=coord_layers[i], ymax=coord_layers[i+1], color=colors[i], linestyle='dashed', linewidth=ln_width, label='_nolegend_')
        plt.axis('off')
        lineL, = plt.plot(np.ones([len(OCT),1]), color='r', linestyle='dashed', linewidth=1, label='Segmentation totale du stroma')
        #lineL.set_label('Segmentation totale du stroma') 
        #plt.legend(lineL,'Segmentation totale du stroma', fontsize=11, loc = 'lower right')  
        lineL.set_visible(False)   
    

#%% 3.2 Création du masque de correction
if coord_centre_img == 1:
    coord_centre = int(np.shape(OCT)[1]/2)
    
if corr == True:
    #Mean_I_layers_smooth = sgn.savgol_filter(Mean_I_layers, dict['w_filterLayers'], sgolay_order)
    
    #from scipy.signal import savgol_coeffs
    #from scipy.special import factorial
    #windowlen = dict['w_filterLayers']
    #order = sgolay_order
    #Mean_I_layers_smooth = np.array([savgol_coeffs(windowlen, order, deriv=d, use='dot') for d in range(order+1)]).T / factorial(np.arange(order+1))
    
        # Délimitation de l'artefact : ACP n°1 ####################################
    X = StandardScaler(with_std=False).fit_transform(Mean_I_layers)    # Données centrées/normées (standardisées)
    pca1 = PCA(.98)          # PCs conservées tq 98% de la variance soit expliquée
    x_new = pca1.fit_transform(X)
    print('variance expliquée PCA1/PC2 puis PCA2/PC1 [%] :')
    print(100*pca1.explained_variance_ratio_)
    
    # ==================> Sur MALTAB PC1 89% vs 59% ici : manquait smooth => 82%...??
    
    nPC_1 = np.shape(pca1.explained_variance_ratio_)[0]
    coeff1 = np.transpose(pca1.components_[0:nPC_1, :])      # vecteurs propres
    score1 = x_new[:,0:nPC_1]                               # valeurs propres
    
        # 2ème CP : délimitation de l'artefact
    PC2 = np.outer(score1[:,1],coeff1.T[1,:]) #============================================================> ACP légèrement différente sur MATLAB
    std_PC2 = sgn.savgol_filter(np.std(PC2,axis=0),15,2)
    std_PC2_smooth = sgn.savgol_filter(np.std(PC2,axis=0),75,2)
    centre_std2 = np.where(np.abs(PC2[-1,]) == np.amax(np.abs(PC2[-1,])))[0][0]    
    ind = detect_peaks(std_PC2_smooth, mph1, mpd1, threshold1, edge='falling', kpsh=False, valley=True, show=show, ax=None, title=True)
    temp = ind[ind > centre_std2 + marge_centre]
    if len(temp) != 0:
        coord_art_max = np.amin(temp)
    else:
        coord_art_max = len(std_PC2_smooth)
    temp = ind[ind < centre_std2 - marge_centre]
    if len(temp) != 0:
        coord_art_min = np.amax(temp)
    else:
        coord_art_min = 0
#    coord_max = np.where(std_PC2 == np.amax(std_PC2[coord_centre2-200:coord_centre2+200]))[0][0]
#    coord_art_max = np.amin(ind[ind > coord_max+marge_centre])
#    coord_art_min = np.amax(ind[ind < coord_max-marge_centre])
    
        # Affichage CP2 + artefact
    plt.figure(11)
    ax = plt.axes()
    ax.set_prop_cycle('color', colors)
    plt.plot(PC2.T, linewidth = 1)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axvline(coord_art_min, ymin, ymax, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(coord_art_max, ymin, ymax, color='r', linestyle='dashed', linewidth=1)
    plt.title('Délimitation de la zone artefactuelle')
    plt.xlabel('Coordonnée latérale [px]', fontsize=fontsize_label)
    plt.ylabel('Valeurs propres de la 2ème composante principale', fontsize=fontsize_label)
    if save == True:
    		plt.savefig(pathname + '\\3_corr_delim_artefact.png', dpi=fig_dpi, bbox_inches='tight')
            
    # INSÉRER GUI : activer la correction ??
    
        # Magnitude de la correction par strate : ACP n°2 ############################
    # "Centrage" des données
    Mean_I_artefact = Mean_I_layers[:,coord_art_min:coord_art_max]
    sz_artefact = np.shape(Mean_I_artefact)
    Mean_I_artefact_offset = np.zeros(np.shape(Mean_I_artefact))
    if np.std(Mean_I_artefact[1:20,:]) < np.std(Mean_I_artefact[:,sz_artefact[1]-20:sz_artefact[1]]):
        for i in range(0,N_layers):
            Mean_I_artefact_offset[i,:] = Mean_I_artefact[i,:] - min(Mean_I_artefact[i,1:20])
    else:
        for i in range(0,N_layers):
            Mean_I_artefact_offset[i,:] = Mean_I_artefact[i,:] - min(Mean_I_artefact[i,sz_artefact[1]-20:sz_artefact[1]])
    # ACP
    X = StandardScaler(with_std=False).fit_transform(Mean_I_artefact_offset)    # Données centrées/normées (standardisées)
    pca2 = PCA(0.995)          # PCs conservées tq 99.5% de la variance soit expliquée
    x_new = pca2.fit_transform(X)
    print(100*pca2.explained_variance_ratio_)
    nPC_2 = np.shape(pca2.explained_variance_ratio_)[0]
    coeff2 = np.transpose(pca2.components_[0:nPC_2, :])      # vecteurs propres
    score2 = x_new[:,0:nPC_2]                                # valeurs propres
    score2_PC1 = score2[:,0]
    score2_PC1_smooth = sgn.savgol_filter(score2_PC1, int(2*np.around(score2_PC1.shape[0]/2)-1), 2)
    xp = coord_layers[:-1] + 3
    score2_PC1_interp = np.interp(range(profondeur_Bowman, profondeur_endoth), xp, score2_PC1_smooth, left=None, right=None, period=None)
    # Reconstruction
    PC1 = np.outer(score2_PC1_interp,coeff2.T[0,:]) #===============================================> ATTENTION coeff *10 pour img8 pour faire comme dans MATLAB
#    PC2 = np.outer(score2[:,1],coeff2.T[1,:])
#    PC3 = np.outer(score2[:,2],coeff2.T[2,:])    
    
        # Définition du masque ####################################################
    # Ajustement de la correction
    PC1_base = PC1 - PC1[1,:]
    PC1_base[PC1_base<0] = 0                # Interdit de réhausser la luminosité
    # Interpolation des valeurs propres de la PC1 (2e ACP)
#    for i in range(np.shape(coord_layers)[0]-1):
#        thk = coord_layers[i+1] - coord_layers[i]
#        CP1_int = np.zeros((thk,sz_artefact[1]))
#        for j in range(sz_artefact[1]-1):
#            CP1_int[:,j] = np.linspace(PC1_base[i,j],PC1_base[i,j+1],thk).T
#        if i == 0:
#            CP1_ext = CP1_int
#        else:
#            CP1_ext = np.vstack((CP1_ext,CP1_int))
    
    # Masque
    mask_offset = 1.1*np.max(PC1_base)
    mask = mask_offset*np.ones((profondeur_endoth-profondeur_Bowman,colonne_sup-colonne_inf))
    mask[:,coord_art_min:coord_art_max] = mask[:,coord_art_min:coord_art_max] - PC1_base
#    for j in range(sz_artefact[1]):
##        mask[:,coord_art_min + j] = mask[:, coord_art_min + j] - CP1_ext[:,j]
#        mask[:,coord_art_min + j] = mask[:, coord_art_min + j] - 
#        
    # Correction
    ROI = deepcopy(FlattenedImage2[profondeur_Bowman:profondeur_endoth,colonne_inf:colonne_sup])
    newROI = deepcopy(ROI) + mask - mask_offset
    
    plt.figure(12)
    plt.subplot(311)
    plt.imshow(mask-mask_offset,cmap = 'gray')
    plt.title('Mask')
    plt.axis('on')
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(ROI,cmap = 'gray')
    plt.title('ROI brute')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(newROI,cmap = 'gray')
    plt.title('ROI corrigée')
    plt.axis('off')
    plt.colorbar()
    if save == True:
    		plt.savefig(pathname + '\\4_corr_masque.png', dpi=fig_dpi, bbox_inches='tight')
    
    # Contrôle
    Mean_I_layers_ctrl = np.zeros(np.shape(Mean_I_layers))
    for i in range(0,len(coord_layers)-1):
        if w_filterLayers != None:      # Lissage
            Mean_I_layers_ctrl[i,:] = sgn.savgol_filter(np.mean(newROI[coord_layers[i]-coord_layers[0]:coord_layers[i+1]-coord_layers[0],:], axis=0), w_filterLayers, 2)
        else:                           # Pas de lissage
            Mean_I_layers_ctrl[i,:] = np.mean(newROI[coord_layers[i]-coord_layers[0]:coord_layers[i+1]-coord_layers[0],:], axis=0)
            
    plt.figure(8)
    
    plt.clf()
    plt.subplot(121)
    for i in range(N_layers):
        label_str = "%i" %i
        plt.plot(Mean_I_layers[i,:].T, linewidth=1, c=colors[i], label=label_str)
#    plt.legend(title='N°', fontsize = 'x-small', loc='center left', bbox_to_anchor=(0.95, 0.5))
    xmin, xmax, ymin_2, ymax_2 = plt.axis()    
    plt.axvline(coord_art_min, ymin, ymax, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(coord_art_max, ymin, ymax, color='r', linestyle='dashed', linewidth=1)
#    plt.title('Avant correction')
#    plt.xlabel('Coordonnée latérale [px]', fontsize=fontsize_label)
#    plt.ylabel('Intensité OCT moyenne brute', fontsize=fontsize_label)
    plt.title('Before correction')
    plt.xlabel('Lateral coordinate [px]', fontsize=fontsize_label)
    plt.ylabel('Raw mean OCT intensity', fontsize=fontsize_label)
    
    ax = plt.subplot(122)
    for i in range(N_layers):
        label_str = "%i" %i
        plt.plot(Mean_I_layers_ctrl[i,:].T, linewidth=1, c=colors[i], label=label_str)
    plt.legend(title='layer', fontsize = 'x-small', loc='center left', bbox_to_anchor=(0.95, 0.5))
    plt.axvline(coord_art_min, ymin, ymax, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(coord_art_max, ymin, ymax, color='r', linestyle='dashed', linewidth=1)
    ax.set_ylim((ymin_2,ymax_2))
#    plt.title('Après correction')
#    plt.xlabel('Coordonnée latérale [px]', fontsize=fontsize_label)
#    plt.ylabel('Intensité OCT moyenne brute', fontsize=fontsize_label)
    plt.title('After correction')
    plt.xlabel('Lateral coordinate [px]', fontsize=fontsize_label)
    plt.ylabel('Raw mean OCT intensity', fontsize=fontsize_label)
    plt.show()
    if save == True:
    		plt.savefig(pathname + '\\4_corr_comparaison_%i_layers.png' %N_layers, dpi=fig_dpi, bbox_inches='tight')
  
    #%% Créer image corrigée
    CorrImage = deepcopy(FlattenedImage2)
    CorrImage[profondeur_Bowman:profondeur_endoth,colonne_inf:colonne_sup] = deepcopy(newROI)
#    plt.figure()
#    plt.imshow(CorrImage-FlattenedImage2,cmap='gray')
#    plt.colorbar()
#    np.any(FlattenedImage2!=CorrImage)
#    np.any(newROI!=ROI)
 
#%% Comparaison pré- et post-correction
if corr == True:
        
    depth_ROI = np.shape(np.mean(ROI, axis=1))[0]
    width_ROI = np.shape(np.mean(ROI, axis=0))[0]
    x = range(depth_ROI)
    yerr1 = np.std(ROI, axis=1)/np.sqrt(width_ROI-1)
    yerr2 = np.std(newROI, axis=1)/np.sqrt(width_ROI-1)
    
    plt.figure(13)
    plt.clf()
    plt.errorbar(x, np.mean(ROI, axis=1), yerr1, label='avant', color='k', linewidth =1, capsize = 2)
    plt.errorbar(x, np.mean(newROI, axis=1), yerr2, label='après',color='r', linewidth = 1, capsize = 2)
    plt.xlim(0,depth_ROI-1)
    ymin1, ymax1 = plt.ylim()    
    plt.ylim(ymin1,ymax1)
    lineW, = plt.plot(np.ones([depth_ROI,1]), color='r', linestyle='dashed', linewidth=1, label='marges')
    #plt.legend(('Avant','Après'), title='n° CP', fontsize = 'x-small', loc='upper left', bbox_to_anchor=(.92, 1))
    #plt.legend(title='n° CP', fontsize = 'x-small', bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.title('Comparaison des profils d\'atténuation avant et après correction')
    plt.xlabel('Profondeur dans le stroma [px]', fontsize=fontsize_label)
    plt.ylabel('Intensité OCT moyenne brute', fontsize=fontsize_label)
    plt.legend(loc='upper right', title='Légende', fontsize='x-small')
    plt.vlines(profondeur_min-profondeur_Bowman, ymin=ymin1, ymax=ymax1, color='r', linestyle='dashed', linewidth=1)
    plt.vlines(profondeur_max-profondeur_Bowman, ymin=ymin1, ymax=ymax1, color='r', linestyle='dashed', linewidth=1)
    plt.show()
    lineW.set_visible(False)
    
    if save == True:
        plt.savefig(pathname + '\\4_corr_comparaison_profils.png', dpi=fig_dpi, bbox_inches='tight')

#for i in range(nPC_2):
##    lineT = []
#    lineW = 0.5 + 3*pca1.explained_variance_ratio_[i]
##    lineT.append(plt.plot(coeff1[:,i], linewidth = lineW))
#    label_str = "%0d: %.1f %%" %(i+1, 100*pca2.explained_variance_ratio_[i])
#    plt.plot(coeff1[:,i], linewidth = lineW, label = label_str)
##plt.legend(lineT, ["%0d: %.1f %%" %(i, 100*pca2.explained_variance_ratio_[i]) for i in range(nPC_2)], title='n° CP', fontsize = 'x-small', loc='upper left', bbox_to_anchor=(.92, 1))
#plt.legend(title='n° CP', fontsize = 'x-small', bbox_to_anchor=(1.05, 1), loc='upper right')
#plt.title('Vecteurs propres de l\'ACP n°1')
#plt.xlabel('Coordonnée latérale [px]', fontsize=fontsize_label)
#plt.ylabel('Intensité OCT moyenne brute du vecteur propre', fontsize=fontsize_label)
#plt.show()

#%% 3.3  Normalisation latérale "géométrique" de l'image + paramètre de saturation de l'épithélium
# Par la valeur lissée de l'intensité des pixels de l'épithelium, correspondant à un maximum de réflexion.
         
if norm == 1:
    #filter_orderNorm = 15                      # largeur de la fenêtre glissante du filtre médian =====================> à justifier
    w_filterNorm = dict['w_filterNorm']         # largeur du filtre de SG (doit être impair)
    sgolay_orderNorm = sgolay_order             # ordre du filtre de SG    
    M = FlattenedImage2[:max_window[1], :]
    VectMax = M.max(axis=0)
#	VectNorm = sgn.medfilt(VectMax, filter_orderNorm)    # ==========================================================> encore très bruité !
    VectNorm = sgn.savgol_filter(VectMax, w_filterNorm, sgolay_orderNorm)
    VectNorm[VectNorm<=75] = 75
    if corr == False:
        ProcessedImage = deepcopy(FlattenedImage2/VectNorm)
    else:
        ProcessedImage = deepcopy(CorrImage/VectNorm)    
    VectNorm_sat = np.where(VectMax==255)[0]
    quant = 0.05
    if VectNorm_sat.shape[0] != 0:        
        epith_sat_min = int(np.quantile(VectNorm_sat,quant))
        epith_sat_max = int(np.quantile(VectNorm_sat,1-quant))
    else:
        epith_sat_min = coord_centre
        epith_sat_max = coord_centre
    epith_sat_um = int((epith_sat_max-epith_sat_min)*pas_lat)
    plt.figure(14)
    lineM, = plt.plot(VectMax, linewidth=0.8)    
    lineN, = plt.plot(VectNorm, linewidth=2)   
    plt.legend((lineM, lineN),('Max(épithélium)','Lissage passe-bas'), fontsize=14)   
    plt.xlabel('Pixels latéraux', fontsize=fontsize_label)   
    plt.ylabel('Intensité OCT moyenne [8-bit]', fontsize=fontsize_label)
#    (ymin,ymax) = ax.get_ylim()
#    plt.ylim(ymin,ymax)
    plt.text(0,np.max(VectMax-5),'Largeur saturée = %i µm'%epith_sat_um,horizontalalignment='left',fontsize=12)
    plt.plot(tuple((epith_sat_min,epith_sat_max)),tuple((VectNorm[epith_sat_min],VectNorm[epith_sat_max])), '+', mfc=None, mec='b', mew=2, ms=8)
    del(M) #, VectMax, VectNorm)
    if 'xmin_cut' in locals():
        if xmax_cut > xmin_cut+5:     # zone spéculaire coupée si intervalle > 5
            plt.title('Courbe de normalisation par l\'épithélium (cut=ON)')              
            if save == True:
                plt.savefig(pathname + '\\5_normVector_SpecularCutON.png', dpi=fig_dpi, bbox_inches='tight')
        else:
            plt.title('Courbe de normalisation par l\'épithélium (cut=OFF)') 
            if save == True:
                plt.savefig(pathname + '\\5_normVector_SpecularCutOFF.png', dpi=fig_dpi, bbox_inches='tight')
    else:
        plt.title('Courbe de normalisation par l\'épithélium (cut=OFF)')
        if save == True:
            plt.savefig(pathname + '\\5_normVector_SpecularCutOFF.png', dpi=fig_dpi, bbox_inches='tight')
    if save == True:
        plt.imsave(pathname + '\\5_normImage.png', ProcessedImage, cmap='gray', dpi=fig_dpi)
        
    fig = plt.figure(15)
    ax = fig.add_subplot(211, facecolor='#FFFFCC')
    plt.imshow(FlattenedImage2[:profondeur_max+100,:], cmap="gray")    
    plt.hlines(profondeur_min, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
    plt.hlines(profondeur_max, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
    plt.vlines(colonne_inf, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
    plt.vlines(colonne_sup, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
    plt.title('Image aplatie brute')
    plt.axis('off')    
    
    ax = fig.add_subplot(212, facecolor='#FFFFCC')
    plt.imshow(ProcessedImage[:profondeur_max+100,:], cmap="gray")    
    plt.hlines(profondeur_min, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
    plt.hlines(profondeur_max, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
    plt.vlines(colonne_inf, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
    plt.vlines(colonne_sup, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
    plt.title('Image aplatie : corr=%s | norm=%s' %(corr,norm))
    plt.axis('off')
    
    if save == True:
            plt.savefig(pathname + '\\6_résumé_OCT.png', dpi=fig_dpi, bbox_inches='tight')
        
elif norm == 0:
    if corr == False:
        ProcessedImage = FlattenedImage2[:]
        if save == True:
            plt.imsave(pathname + '\\5_NonNormNonCorrImage.png', ProcessedImage, cmap='gray', dpi=fig_dpi)
    else:
        ProcessedImage = CorrImage[:]
        if save == True:
            plt.imsave(pathname + '\\5_NonNormCorrImage.png', ProcessedImage, cmap='gray', dpi=fig_dpi)

if corr == True:
    fig_aff = 5
    plt.figure(fig_aff)
    plt.show()
else:
    fig_aff = 15
#del(FlattenedImage)# FlattenedImage2)    


#%% 4/ EXPORT ////////////////////////////////////////////////////////////////
CutImage = ProcessedImage[profondeur_min:profondeur_max,colonne_inf:colonne_sup]
Mean_I = np.mean(CutImage, axis=1)
std_I = np.std(CutImage, axis=1)/np.sqrt(np.shape(CutImage)[1]-1)
# incertitude sur la moyenne = écart-type/ racine(nbre de pixel)

N_pix_depth_short = np.shape(Mean_I)[0]
z_stroma = np.arange(0, N_pix_depth_short*pas, pas)[:N_pix_depth_short]
Data = np.column_stack((z_stroma.T, Mean_I, std_I))
header = ['tag0','SNR_2D','ep_cornee_um','ep_stroma_tot_um','ep_stroma_analyse_um','SpecularCut_µm','champ_mm','epith_sat_um','seuil_epith','prof_Bowman_px','prof_epith_px','marge_postBowman_µm','marge_preDescemet_µm','pas_ax','pas_lat']
lines = [Path_patient,SNR_2D,ep_cornee_um,ep_stroma_tot_um,ep_stroma_analyse_um,SpecularCut,champ_analyse_mm,epith_sat_um,seuil,profondeur_Bowman,profondeur_endoth,marge_postBowman,marge_preDescemet,pas,pas_lat]

if corr == True and save == True:
#        plt.savefig(pathname + '\\3_corr_StromalLayers_%i_profil.png' %N_layers)
        np.savetxt(pathname + '\\StromalLayers_%i_courbes.csv' %N_layers, Mean_I_layers, delimiter=",")

if save == True:
    with open(pathname + '/dataPreprocess.csv', "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header) # write the header
#        for l in lines:
#            writer.writerow(l)
        writer.writerow(lines)
#        writer.writerows(lines)
    
if save == True:
    np.savetxt(pathname + '/Courbe.csv', Data, delimiter=",")
    numbers = open(pathname + '/dataPreprocess.txt', 'a')
    numbers.truncate(0)
    numbers.write('SNR_2D = %.1f dB\n' % SNR_2D)
    numbers.write('gauss_sigma = %i\n' % gauss_sigma)
    if norm == 1:
        numbers.write('norm = True_épith\n')
    elif norm == 2:
        numbers.write('norm = True_stroma_ant\n')
    elif norm == 0:
        numbers.write('norm = False\n')
    if SpecularCut_option == 1:
        numbers.write('SpecularCut_option = True_MANUELLE\n')
    elif SpecularCut_option == 2:
        numbers.write('SpecularCut_option = True_FIXE\n')
    elif SpecularCut_option == 3:
        numbers.write('SpecularCut_option = True_AUTO\n')
    elif SpecularCut_option == 0:
        numbers.write('SpecularCut_option = True_SPECIFIED_BY_USER\n')
    if seg_stroma == 1:
        numbers.write('seg_stroma = AUTO\n') 
    elif seg_stroma == 0:
        numbers.write('seg_stroma = MANUELLE\n')           
    numbers.write('corr = %s\n' % corr)        
    numbers.write('pas_axial = %.3f µm\n' % pas)
    numbers.write('epaisseur_cornee_um = %i µm\n' % ep_cornee_um)
    numbers.write('epaisseur_stroma_tot_um = %i µm\n' % ep_stroma_tot_um)
    numbers.write('epaisseur_stroma_analyse_um = %i µm\n' % ep_stroma_analyse_um)
#    if coord_centre_img == 1:
#        numbers.write('coord_centre_img = None\n')
#    else:
    numbers.write('coord_centre_img = %i px\n' % coord_centre2) 
    numbers.write('specular_cut = %d µm\n' % SpecularCut)
    numbers.write('champ_analyse = %i µm\n' % int(SpecularCut+(colonne_sup-colonne_inf)*pas_lat))
    numbers.write('epith_sat_um = %i um\n'%epith_sat_um)
    numbers.write('quantile_sat_det = %.3f\n' % quant)
    numbers.write('colonne_inf = %d\n' % colonne_inf)
    numbers.write('colonne_sup = %d\n' % colonne_sup)
    numbers.write('profondeur_Bowman = %d\n' % profondeur_Bowman)
    numbers.write('profondeur_endoth = %d\n' % profondeur_endoth)
    numbers.write('marge_postBowman = %d µm\n' % marge_postBowman)
    numbers.write('marge_preDescemet = %d µm\n' % marge_preDescemet)
    if 'marge' in locals():
        if marge != None:
            numbers.write('xmin_cut = %d\n' % xmin_cut)
            numbers.write('xmax_cut = %d\n' % xmax_cut)
            numbers.write('marge_SpecCut = %d px\n' % marge)
    numbers.write('z0 = %d\n' % z0)
    numbers.write('seuil = %d\n' % seuil)
    numbers.write('max_window_size = %d px\n' % max_window_size)
    numbers.write('der1_seuil = %0.2f\n' % der1_seuil)
    if corr == True:        
        numbers.write('N_layers = %i\n' % N_layers)      
        numbers.write('profondeur_stroma_large = %d µm\n' % depth_stroma_large)        
        numbers.write('coord_layers = \n%s\n' % np.array2string(coord_layers, max_line_width=float('Inf')))
    if 'w_filterLayers' in locals():
        if w_filterLayers == None:
            numbers.write('lissage: OFF\n')            
        else:           
            numbers.write('w_filterLayers = %i\n' % w_filterLayers)
#    if 'select_depth' not in locals():
#        numbers.write('select_depth: auto')
    numbers.close()
    del champ_analyse_mm

time_end = time.time() 
print('\nEND [t=%s s | t_comp=%.1f s]'%(int(time_end-t0_time), np.round(time_spec_end-time_start+time_flat_end-time_flat_start+time_end-time_corr_start,1)))

#%% Fermer toutes les images
keyboardClick=False
print('=> Appuyez sur ENTRÉE fig.'+ str(fig_aff) +' pour fermer toutes les images (ne pas les fermer à la main, sinon ça buggue).')
while keyboardClick != True:
    keyboardClick=plt.waitforbuttonpress()
    plt.close('all')
    fileindex += 1
    print('\n------------------------------------------------------------------- '+Path_patient+'/'+Path_eye+' : OK [%i/%i = %i%%]'%(fileindex,fileindex_tot,int(100*round(fileindex/fileindex_tot,2))))

#%% RAB

#select_depth = 0        # 0 : sélection épaisseur stroma sur profil intensité, 1 : sur image OCT