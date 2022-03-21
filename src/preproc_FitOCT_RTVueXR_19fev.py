#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=>>> STYLE DE PRESENTATION : https://numpydoc.readthedocs.io/en/latest/format.html <<<<<=
Created on Fri Dec  3 14:03:53 2021

@author: Maëlle Vilbert, Romain Bocheux (corneal flattening)

Preprocessing of raw OCT corneal images acquired with RTVue-XR Avanti SD-OCT
device by Optovue (Optovue Inc., Fremont, CA, USA).

Parameters:
    OCT (jpg/png image): do NOT rename the files. You can possibly add
    information at the end of the name (after a _), before the extension. #==========> bonne idée de garder le nom ou pas ? (anonymisation à l'export des 15-20)

Returns:
    The mean stromal in-depth intensity profile, to be used as an input in
    Pascal Pernot's FitOCT program: https://doi.org/10.5281/zenodo.2579915

"""
#%% ENVIRONMENT
#os.chdir('C:\\Users\\Maëlle\\Documents\\Codes - FitOCTlib\\Python')

# Imports
import os
import glob
import unicodedata
import tkinter as tk
from copy import deepcopy
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sgn
import scipy.ndimage as img
    # Custom functions
os.chdir('C:\\Users\\Maëlle\\Documents\\Codes - FitOCTlib\\Python')
from preproc_functions import SNR, histogram_shift, saturation_artifact_cut, anterior_surface_detection_and_flattening, stromal_z_segmentation, stromal_x_boundaries, stromal_sublayers, posterior_stromal_artifact_localization, posterior_stromal_artifact_mask, geometrical_normalization

# Graphical parameters (plt)
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['figure.figsize'] = 14.5, 8                # figure size [inch]
fontsize_title = 20                                     # title font size
fontsize_label = 15                                     # axis label font size
plt.rcParams.update({'font.size': fontsize_title})
plt.rcParams.update({'xtick.labelsize': 'xx-small'})    # x-tick font size
plt.rcParams.update({'ytick.labelsize': 'xx-small'})    # y-tick font size
plt.rcParams.update({'axes.labelsize': 'small'})
fig_dpi = 150                                           # resolution of exported figures

# Clock
t0_time = time.time()

#%% USER PARAMETERS

pathname = 'C:\\Users\\Maëlle\\Documents\\Rédaction\\_Article in vivo\\code' # ==> nom: dirpath (pour directory path)
extension = '.jpg'       
filename = [os.path.basename(x) for x in glob.glob(os.path.join(pathname, '*'+ extension))][6]

champ_analyse_mm = 6        # [mm] width of analyzed area (comment the line if analyzing the full image width) =====> nom: roi_width_mm
marge_postBowman = 60       # [µm] post-Bowman layer z-margin [default 60 µm] ======================================> nom: margin_postBowman
marge_preDescemet = 30      # [µm] pre-Descemet membrane z-margin [default 30 µm] ==================================> nom: margin_preDescemet

save = False                # save figures and .txt/.csv files (bool)  
corr = True                 # compute posterior artefact correction (bool)
#show = False                 # display additional figures (bool)
user_validation = False      # ask user to validate segmentation results (bool)

#%% AUTOMATED PREPROCESSING ALGORITHM
#%% (A.1) Initialization
plt.close('all')
time_start = time.time()

# Default dictionary: values chosen for RTVue-XR Avanti SD-OCT images.
dict = {}
dict.setdefault('max_window_size', 10) #==========================================> nom: anterior_surface_z_window [px]
dict.setdefault('profondeur_min', None) #=========================================> nom: stroma_z_top
dict.setdefault('profondeur_max', None) #=========================================> nom: stroma_z_bottom
dict.setdefault('median_filter_width',15) #=======================================> nom: median_filter_kernel_size
dict.setdefault('w_filter',101) #=================================================> nom: savgol_filter_window
dict.setdefault('sgolay_order',2)
dict.setdefault('w_filterNorm',115) #=============================================> nom: savgol_filter_window_norm
dict.setdefault('nLayers', 20) #==================================================> nom: n_layers
dict.setdefault('w_filterLayers',275) # int ou None ==============================> nom: savgol_filter_window_layers

gauss_sigma = 1      # lissage gaussien 2D (SNR, speccut) ========================> nom: gaussian_filter_sigma

# Metadata import
filename_split = filename.split('_')
    # Patient code name (without accent)
Path_patient = filename_split[1][:3].upper() + filename_split[2][:3].upper() # ================> nom: patient_name
Path_patient = unicodedata.normalize('NFD', Path_patient).encode('ascii', 'ignore').decode("utf-8")
Path_eye = filename_split[5] #=================================================================> nom: mode_acquisition
if 'Line' in Path_eye: #===========================================================> ? en faire une fonction ?
    dict.setdefault('champ_acquisition_mm', 8)          # FOV = 8 mm ==============> nom: FOV_x_mm
    pas = 4.333                                         # [µm] axial pixel size ===> nom: pixel_size_z
    if 'Cross' in Path_eye:
        Path_eye = 'Cross'
    else:
        Path_eye = 'Line'
elif 'Pachymetry' in Path_eye:
    if 'Wide' in Path_eye:
        Path_eye = 'PachyWide'
        dict.setdefault('champ_acquisition_mm', 9)      # FOV = 9 mm
        pas = 4.322                                     
    else:
        Path_eye = 'Pachy'
        dict.setdefault('champ_acquisition_mm', 6)      # FOV = 6 mm
        pas = 4.333
#FOV_x_px = np.round(dict['champ_acquisition_mm']*1000/pas,2) #↨ ================> pas besoin ?

# Raw OCT image import
X = mpimg.imread(os.path.join(pathname,filename))
z0 = 10                                 # image top-marging width [px]
if extension == '.jpg':
    OCT_brut = X[z0:, 2:, 0]*1.0                # conversion into float64 #=========> OCT
elif extension == '.jpeg':
    OCT_brut = X[z0:, 2:]*1.0                   # conversion into float64 #=========> OCT

pas_lat = round(1000*dict['champ_acquisition_mm']/OCT_brut.shape[1],2) # [µm] lateral pixel size ===> nom: pixel_size_x
      
# Deletion of the "orientation arrow" on raw images
sz_arrow = 70 #=================================================================> nv nom = ARROW_SIZE
OCT_brut[0:sz_arrow,np.shape(OCT_brut)[1]-sz_arrow:np.shape(OCT_brut)[1]] = np.zeros((sz_arrow,sz_arrow)) # remove the scan orientation arrow

# Computation of bidimensional SNR
SNR_2D = SNR(OCT_brut)
print('SNR_2D = %.1f dB'%SNR_2D)

#%% (A.2) Exposure correction of overexposed images

OCT_brut = histogram_shift(OCT_brut, show=True)
    
SNR_2D = SNR(OCT_brut)
print('SNR_2D (exposure adjusted) = %.1f dB'%SNR_2D)

#%% (B) Saturation artifact detection

coord_centre = None #============================================================> nom: apex_x_coordinate
if 'champ_analyse_mm' not in locals():
    champ_analyse_mm = dict['champ_acquisition_mm']

auto = True
coord_centre, xmin_cut, xmax_cut, SpecularCut, derivative1_smooth = saturation_artifact_cut( #========> nom: SpecularCut -> saturation_cut_width
        OCT_brut, pas_lat, auto) #====================================================================> nom: xmin_cut => xmin_saturation_cut, idem pour xmax_cut

time_spec_end = time.time()
###############################################################################
############################### User approval #################################
###############################################################################

if user_validation:    
    result = None
    temp = None
    temp_exit = None
    root = tk.Tk()
    canvas2 = tk.Canvas(root, width = 300, height = 0)
    canvas2.pack()
    
    def CutOK():
        root.destroy()
    def CutNotOK():
        MsgBox = tk.messagebox.askokcancel('Removal rejected', 'Manual cut?', default='ok', icon = 'question') # cancel button in French...
        if MsgBox:
            global result
            plt.figure(2)
            plt.close()
            auto = False
            result = saturation_artifact_cut(OCT_brut, pas_lat, auto)
            return result, auto
        else:
            auto = False
            OCT = img.gaussian_filter(OCT_brut, sigma = gauss_sigma) # =================> OCT: OCT_smooth
            mean_signal = np.mean(OCT, axis = 0)
            savgol_window = 15
            savgol_order = 2
            signal_smooth = sgn.savgol_filter(mean_signal, savgol_window, savgol_order)
            derivative1 = np.diff(signal_smooth)
            plt.figure(2)
            plt.clf()
            plt.imshow(OCT_brut,cmap = 'gray')
            offset_aff = np.ceil(np.amax(mean_signal)) + 5
            lineN, = plt.plot(offset_aff + -signal_smooth, color='tab:purple')
            lineO, = plt.plot(offset_aff + -10*derivative1, color='tab:green')
            plt.title("Artifact removal (manual) : width = 0 µm")
            plt.legend((lineN, lineO),('Smoothed mean x-signal','First derivative of x-signal',), fontsize=11, loc = 'lower right') 
            plt.show()
            global temp
            temp = 1
            root.destroy()
            return temp, auto
    def ExitApp():
        root.destroy()
        raise SystemExit(...)
    
    button1 = tk.Button(root, text='Removal accepted',command=CutOK,bg='green',fg='white').pack(fill=tk.X)
    canvas2.create_window(150, 50, window=button1)
    button2 = tk.Button(root, text='Removal rejected: manual mode',command=CutNotOK,bg='brown',fg='white').pack(fill=tk.X)
    canvas2.create_window(150, 100, window=button2)
    button3 = tk.Button(root, text='Terminate execution',command=ExitApp,bg='red',fg='white').pack(fill=tk.X)
    canvas2.create_window(150, 150, window=button3)
      
    root.mainloop()
    
    coord_centre_img = 0
    if result is not None:
        xmin_cut = result[1]
        xmax_cut = result[2]
        coord_centre = result[0]
        SpecularCut = result[3] #=======================================================> nom: saturation_cut_width
        coord_centre_img = 0
        derivative1_smooth = result[4]
    
    if temp is not None:
        SpecularCut = 0 #==============================================================> nom: saturation_cut_width
        xmin_cut = 0
        xmax_cut = 0 
            
    if save:
        if not auto:
            fig = plt.figure(2)
            plt.savefig(pathname + '\\1_SaturationCut_manual_%ium.png'%int(np.round(SpecularCut)), dpi=fig_dpi, bbox_inches='tight')
        else:
            fig = plt.figure(2)
            plt.savefig(pathname + '\\1_SaturationCut_auto_%ium.png'%int(np.round(SpecularCut)), dpi=fig_dpi, bbox_inches='tight')
    
    del button1, button2, button3, result, temp, temp_exit
    
###############################################################################      
# OCT image cutting
if 'xmin_cut' in locals():
    OCT_cut_brut = np.delete(OCT_brut, range(xmin_cut, xmax_cut), axis=1) #========> nom: OCT_cut (ATTENTION : renommer d'abord OCT_cut en OCT_cut_smooth !!)


 
#%% (C) Anterior surface detection and (D.1) corneal flattening
time_flat_start = time.time() 

max_window_size = dict['max_window_size']   # z-window (px) for air-tear interface detection. ===============> nom: anterior_surface_z_window
stroma_z_anterior_bound = dict['profondeur_min']     # z-coordinate (px) of the anterior boundary of the stroma on flattened images [default None].
stroma_z_posterior_bound = dict['profondeur_max']     # z-coordinate (px) of the posterior boundary of the stroma on flattened images [default None].
w_filter = dict['w_filter']                         # length (px) of the Savitzky-Golay filter window (must be less than OCT_cut.shape[1]) [default 101].
median_filter_width = dict['median_filter_width']   # size (px) of the median filter window.
sgolay_order = dict['sgolay_order']                 # order of the polynomial used to fit the samples (must be less than w_filter) [default 2].

#OCT_cut = img.gaussian_filter(OCT_cut_brut, sigma=2) #======================================================> nom: OCT_cut_smooth USELESS
anterior_surface_z_coordinate, FlattenedImage, seuil = anterior_surface_detection_and_flattening(OCT_cut_brut, SpecularCut, xmin_cut,
                               xmax_cut, derivative1_smooth, Path_eye, 
                               method='threshold', max_window_size=10, gauss_sigma=1, median_filter_width=15, 
                               w_filter=101, sgolay_order=2, show=True) #====================================> nom: FlattenedImage => flattened_OCT

columns = np.shape(OCT_cut_brut)[1] #========================================================================> nom: x_dim
depth = np.shape(OCT_cut_brut)[0] # =========================================================================> nom: z_dim

coord_centre2 = int(np.where(anterior_surface_z_coordinate == np.amin(anterior_surface_z_coordinate[100:columns-100]))[0][0]) # ==========> nom: apex_x_coordinate_refined

# Refined flattening
anterior_surface_z_coordinate = anterior_surface_z_coordinate - min(anterior_surface_z_coordinate)
offset = int(round(min(anterior_surface_z_coordinate))) # global z-offset before corneal apex

anterior_surface_z_coordinate_refined, FlattenedImage2, _ = anterior_surface_detection_and_flattening(FlattenedImage, SpecularCut, xmin_cut, 
                                                                                                   xmax_cut, derivative1_smooth, Path_eye,
                                                                                                   method='z_max_in_window', max_window_size=10,
                                                                                                   gauss_sigma=1, median_filter_width=15,
                                                                                                   w_filter=101, sgolay_order=2, show=False) #==========> nom: flattened_OCT_refined

#%% (D.2) Stromal segmentation
show = False
# In-depth (z-) segmentation of the cornea and the ROI
coord_stroma, coord_cornee, ep_cornee_um, ep_stroma_tot_um, profondeur_min, profondeur_max, ep_stroma_analyse_um = stromal_z_segmentation(
        FlattenedImage2, pas, marge_postBowman, marge_preDescemet, min_peak_height=0, min_peak_distance=8, peak_threshold=0.01, peak_edge='both', display_peaks=show)
#=================================================================================> nom: z_coordinates_stroma, z_coordinates_cornea

# Lateral (x-) boundaries of the ROI (delimited by geometrically missing signal on the edges)
saturation_cut_width_px = int(np.round(SpecularCut/pas_lat,0))
champ_acquisition_mm = dict['champ_acquisition_mm'] #============================> en orienté objet = attribut de classe
colonne_inf, colonne_sup = stromal_x_boundaries(
        FlattenedImage2, profondeur_min, profondeur_max, coord_cornee, coord_centre2, champ_acquisition_mm, saturation_cut_width_px, pas_lat, pas, marge_postBowman, marge_preDescemet, pathname, roi_width_mm=champ_analyse_mm, mode='auto')
        # flattened_image, stroma_z_anterior_bound, stroma_z_posterior_bound, z_coordinates_cornea, apex_x_coordinate_refined, FOV_x_mm, saturation_cut_width_px, pixel_size_x, pixel_size_z, margin_postBowman, margin_preDescemet, pathname, MsgBox=np.nan, roi_width_mm=np.nan, mode='auto', save='False')
#=================================================================================> nom: colonne_inf, colonne_sup = stroma_x_left, stroma_x_right

# Figure
CutImage = deepcopy(FlattenedImage2[:int(coord_cornee[1]*1.25), colonne_inf:colonne_sup])
Mean_I_large_adjust = np.mean(CutImage, axis=1)  #==========================================> nom: z_mean_intensity
SD_I_large_adjust = np.std(CutImage, axis=1)#/np.sqrt(np.shape(CutImage)[1]-1) =============> nom: z_sd_intensity

N_pix_depth = np.shape(Mean_I_large_adjust)[0]  #===========================================> nom: z_dim (existe déjà : le même ?)
z = np.arange(0, N_pix_depth*pas, pas)[:N_pix_depth] #======================================> nom: z_axis_px

fig = plt.figure(5)
fig.clf()
ax = fig.add_subplot(211, facecolor='w')
ax.plot(z, Mean_I_large_adjust, linewidth=2)
plt.fill_between(z, Mean_I_large_adjust-SD_I_large_adjust, Mean_I_large_adjust+SD_I_large_adjust,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
    linewidth=0)
segm_x = (coord_cornee[0],coord_stroma[0],coord_stroma[1])
ax.plot(tuple(pas*np.array(segm_x)), Mean_I_large_adjust[(segm_x),], '+', mfc=None, mec='r', mew=2, ms=8) # don't forget the comma for tuple (segm_x) indexing
plt.axvline(profondeur_min*pas, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
plt.axvline(profondeur_max*pas, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
plt.title('In-depth stromal segmentation')
text_str = 'Corneal thickness = %i µm\n Stromal thickness = %i µm (%.1f%%)\n ROI thickness = %i µm (%.1f%%)\n ROI width = %.1f mm' %(ep_cornee_um,ep_stroma_tot_um,100*round(ep_stroma_tot_um/ep_cornee_um,3),ep_stroma_analyse_um,100*round(ep_stroma_analyse_um/ep_cornee_um,3),np.round(((colonne_sup-colonne_inf)*pas_lat+SpecularCut)/1000,1))
(ymin,ymax) = ax.get_ylim()
plt.text(np.amax(z)*1.04,0.7*ymax,text_str,horizontalalignment='right',fontsize=9)
plt.xlabel('Depth [µm]', fontsize=15)
plt.ylabel('Mean OCT intensity [a.u.]', fontsize=15)
    
ax = fig.add_subplot(212)
plt.imshow(FlattenedImage2[:profondeur_max+100,], cmap="gray")
plt.hlines(profondeur_min, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
plt.hlines(profondeur_max, xmin=colonne_inf, xmax=colonne_sup, color='r', linestyle='dashed', linewidth=1)
plt.vlines(colonne_inf, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
plt.vlines(colonne_sup, ymin=profondeur_min, ymax=profondeur_max, color='r', linestyle='dashed', linewidth=1)
plt.axis('off')
plt.show()
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

if save:
    plt.savefig(pathname + '/2_segmentation_stroma_champ_%.1fmm_std.png'%champ_analyse_mm, dpi=fig_dpi, bbox_inches='tight')
    
    # ROI sur profil
#if save:
    fig = plt.figure(6)
    fig.clf()
    ax = fig.add_subplot(111, facecolor='w')
    ax.plot(z, Mean_I_large_adjust, linewidth=2)
    ax.plot(tuple(pas*np.array(coord_stroma)),Mean_I_large_adjust[(coord_stroma),], '+', mfc=None, mec='r', mew=2, ms=8)
    # attention virgule indispensable après tuple pour l'indexation de Mean_I_large
    plt.axvline(profondeur_min*pas, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(profondeur_max*pas, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
    plt.title('In-depth stromal segmentation')
    plt.xlabel('Depth [µm]', fontsize=fontsize_label)
    plt.ylabel('Mean OCT intensity [a.u.]', fontsize=fontsize_label)
    plt.show()
    plt.savefig(pathname + '/2_segmentation_stroma_profil.png', dpi=200, bbox_inches='tight')
    plt.close()
    
time_flat_end = time.time()

###############################################################################
############################### User approval #################################
###############################################################################

if user_validation:    
    result = None
    temp_exit = None
    root = tk.Tk()
    canvas1 = tk.Canvas(root, width = 300, height = 0)
    canvas1.pack()
    
    def SegOK():
        root.destroy()
    def SegNotOK():
        global result    
        MsgBox = tk.messagebox.askquestion('Manual mode', 'Segmentation :  Yes=on the intensity profile / No=on the OCT image', icon = 'question')
    #    fig = plt.figure(5)
        plt.figure(5)
        plt.close()
        if MsgBox:
            result = stromal_x_boundaries(FlattenedImage2, profondeur_min, profondeur_max, coord_cornee, coord_centre2, champ_acquisition_mm, saturation_cut_width_px, pas_lat, pas, marge_postBowman, marge_preDescemet, pathname, MsgBox, roi_width_mm=champ_analyse_mm, mode='manual')
            #flattened_image, stroma_z_anterior_bound, stroma_z_posterior_bound, z_coordinates_cornea, apex_x_coordinate_refined, FOV_x_mm, saturation_cut_width_px, pixel_size_x, pixel_size_z, margin_postBowman, margin_preDescemet, pathname, MsgBox=np.nan, roi_width_mm=np.nan, mode='auto', save='False')
            return result
            print(result)
    
#    def ExitApp():
#        root.destroy()
#        raise SystemExit(...)
    
    button1 = tk.Button(root, text='Segmentation accepted',command=SegOK,bg='green',fg='white').pack(fill=tk.X)
    canvas1.create_window(150, 50, window=button1)
    button2 = tk.Button(root, text='Segmentation rejected: manual mode',command=SegNotOK,bg='brown',fg='white').pack(fill=tk.X)
    canvas1.create_window(150, 100, window=button2)
    button3 = tk.Button(root, text='Terminate execution',command=ExitApp,bg='red',fg='white').pack(fill=tk.X)
    canvas1.create_window(150, 150, window=button3)
      
    root.mainloop()
    
    if result is not None:
        colonne_inf = result[0]
        colonne_sup = result[1]
        
        ep_cornee_um = int(abs(coord_cornee[0]*pas-coord_cornee[1]*pas))
        ep_stroma_tot_um = int(abs(coord_stroma[1]*pas-coord_stroma[0]*pas))
        ep_stroma_analyse_um = int(ep_stroma_tot_um - marge_postBowman-marge_preDescemet)
        
    del(result)


#%% (E.1) Definition of stromal sub-layers
time_corr_start = time.time() 

w_filterLayers = dict['w_filterLayers']
N_layers = dict['nLayers']

coord_layers, Mean_I_layers, colors = stromal_sublayers(N_layers, FlattenedImage2, w_filterLayers, coord_stroma, colonne_inf, colonne_sup, corr='True', save='False', show='True')
#=================================================================================> nom: layers_z_coordinates, layers_mean_x_signal


#%% (E.2) Localization of posterior stromal artifact

mph1 = None                    # minimum peak height [0;1]
mpd1 = 50                      # [px] minimum peak distance 
threshold1 = 0                 # min difference with neighbours [0;1]
marge_centre = 60              # [px] margin around the center of the standard deviation peak.

coord_art_min, coord_art_max = posterior_stromal_artifact_localization(Mean_I_layers, colors, mph1, mpd1, threshold1, marge_centre, pathname, save=False)

#%% (F) PCA-based correction mask

mask, mask_offset, CorrImage = posterior_stromal_artifact_mask(Mean_I_layers, coord_stroma, coord_art_min, coord_art_max, coord_layers, FlattenedImage2, colonne_inf, colonne_sup, w_filterLayers, colors, profondeur_min, profondeur_max, pas, pathname, show=True, save=False)

#%% (G) Lateral normalization

w_filterNorm = dict['w_filterNorm']         # length of the Savitzky-Golay filter window
sgolay_orderNorm = sgolay_order             # order of the polynomial used to fit the samples

input_image = CorrImage
VectNorm, ProcessedImage, epith_sat_um, quant = geometrical_normalization(input_image, coord_stroma, corr, coord_centre, pas_lat, xmin_cut, xmax_cut, profondeur_min, profondeur_max, colonne_inf, colonne_sup, pathname, w_filterNorm, sgolay_orderNorm, save=False)

#%% (H) Signal averaging with depth 

CutImage = ProcessedImage[profondeur_min:profondeur_max,colonne_inf:colonne_sup]
Mean_I = np.mean(CutImage, axis=1)
std_I = np.std(CutImage, axis=1)
sem_I = np.std(CutImage, axis=1)/np.sqrt(np.shape(CutImage)[1]-1)

#%%
time_end = time.time() 
print('\nEND [t=%s s]'%(int(time_end-t0_time)))

if (0):
    plt.close('all')

#%% EXPORT
    
#N_pix_depth_short = np.shape(Mean_I)[0]
#z_stroma = np.arange(0, N_pix_depth_short*pas, pas)[:N_pix_depth_short]
#Data = np.column_stack((z_stroma.T, Mean_I, std_I))
#header = ['tag0','SNR_2D','ep_cornee_um','ep_stroma_tot_um','ep_stroma_analyse_um','SpecularCut_µm','champ_mm','epith_sat_um','seuil_epith','prof_Bowman_px','prof_epith_px','marge_postBowman_µm','marge_preDescemet_µm','pas_ax','pas_lat']
#lines = [Path_patient,SNR_2D,ep_cornee_um,ep_stroma_tot_um,ep_stroma_analyse_um,SpecularCut,champ_analyse_mm,epith_sat_um,seuil,profondeur_Bowman,profondeur_endoth,marge_postBowman,marge_preDescemet,pas,pas_lat]
#
#if corr and save:
#        np.savetxt(pathname + '\\StromalLayers_%i_courbes.csv' %N_layers, Mean_I_layers, delimiter=",")
#
#if save:
#    with open(pathname + '/dataPreprocess.csv', "w", newline='') as f:
#        writer = csv.writer(f, delimiter=',')
#        writer.writerow(header) # write the header
##        for l in lines:
##            writer.writerow(l)
#        writer.writerow(lines)
##        writer.writerows(lines)
#    
#if save:
#    np.savetxt(pathname + '/Courbe.csv', Data, delimiter=",")
#    numbers = open(pathname + '/dataPreprocess.txt', 'a')
#    numbers.truncate(0)
#    numbers.write('SNR_2D = %.1f dB\n' % SNR_2D)
#    numbers.write('gauss_sigma = %i\n' % gauss_sigma)
#    if norm == 1:
#        numbers.write('norm = True_épith\n')
#    elif norm == 2:
#        numbers.write('norm = True_stroma_ant\n')
#    elif norm == 0:
#        numbers.write('norm = False\n')
#    if auto == False:
#        numbers.write('SpecularCut_option = MANUELLE\n')
#    elif auto == True:
#        numbers.write('SpecularCut_option = AUTO\n')
#    if seg_stroma == 1:
#        numbers.write('seg_stroma = AUTO\n') 
#    elif seg_stroma == 0:
#        numbers.write('seg_stroma = MANUELLE\n')           
#    numbers.write('corr = %s\n' % corr)        
#    numbers.write('pas_axial = %.3f µm\n' % pas)
#    numbers.write('epaisseur_cornee_um = %i µm\n' % ep_cornee_um)
#    numbers.write('epaisseur_stroma_tot_um = %i µm\n' % ep_stroma_tot_um)
#    numbers.write('epaisseur_stroma_analyse_um = %i µm\n' % ep_stroma_analyse_um)
##    if coord_centre_img == 1:
##        numbers.write('coord_centre_img = None\n')
##    else:
#    numbers.write('coord_centre_img = %i px\n' % coord_centre2) 
#    numbers.write('specular_cut = %d µm\n' % SpecularCut)
#    numbers.write('champ_analyse = %i µm\n' % int(SpecularCut+(colonne_sup-colonne_inf)*pas_lat))
#    numbers.write('epith_sat_um = %i um\n'%epith_sat_um)
#    numbers.write('quantile_sat_det = %.3f\n' % quant)
#    numbers.write('colonne_inf = %d\n' % colonne_inf)
#    numbers.write('colonne_sup = %d\n' % colonne_sup)
#    profondeur_Bowman = coord_stroma[0]
#    profondeur_endoth = coord_stroma[1]
#    numbers.write('profondeur_Bowman = %d\n' % profondeur_Bowman)
#    numbers.write('profondeur_endoth = %d\n' % profondeur_endoth)
#    numbers.write('marge_postBowman = %d µm\n' % marge_postBowman)
#    numbers.write('marge_preDescemet = %d µm\n' % marge_preDescemet)
#    if 'marge' in locals():
#        if marge != None:
#            numbers.write('xmin_cut = %d\n' % xmin_cut)
#            numbers.write('xmax_cut = %d\n' % xmax_cut)
#            numbers.write('marge_SpecCut = %d px\n' % marge)
#    numbers.write('z0 = %d\n' % z0)
#    numbers.write('seuil = %d\n' % seuil)
#    numbers.write('max_window_size = %d px\n' % max_window_size)
#    numbers.write('der1_seuil = %0.2f\n' % der1_seuil)
#    if corr == True:        
#        numbers.write('N_layers = %i\n' % N_layers)      
#        numbers.write('profondeur_stroma_large = %d µm\n' % depth_stroma_large)        
#        numbers.write('coord_layers = \n%s\n' % np.array2string(coord_layers, max_line_width=float('Inf')))
#    if 'w_filterLayers' in locals():
#        if w_filterLayers == None:
#            numbers.write('lissage: OFF\n')            
#        else:           
#            numbers.write('w_filterLayers = %i\n' % w_filterLayers)
##    if 'select_depth' not in locals():
##        numbers.write('select_depth: auto')
#    numbers.close()
#    del champ_analyse_mm
#

##%% CLOSE FIGURES
#keyboardClick=False
#print('=> Appuyez sur ENTRÉE fig.'+ str(fig_aff) +' pour fermer toutes les images (ne pas les fermer à la main, sinon ça buggue).')
#while keyboardClick != True:
#    keyboardClick=plt.waitforbuttonpress()
#    plt.close('all')
#    fileindex += 1
#    print('\n------------------------------------------------------------------- '+Path_patient+'/'+Path_eye+' : OK [%i/%i = %i%%]'%(fileindex,fileindex_tot,int(100*round(fileindex/fileindex_tot,2))))

