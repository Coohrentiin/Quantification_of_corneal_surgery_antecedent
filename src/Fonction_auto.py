# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:10:34 2020
Fonctions pour l'automatisation du prétraitement de FitOCT

@author: Maëlle
"""
import numpy as np
import matplotlib.pyplot as plt
# pip install detecta
from detecta import detect_peaks # Jupyter notebook @Marcos Duarte
#from detect_peaks import detect_peaks
from matplotlib.widgets import Cursor

def detect_stromal_peaks(Mean_I_large, mph, mpd, threshold, edge='both', aff_peaks = False):
    # Détecte les pics hyper-réflectifs du profil de rétrodiffusion cohérente 
    # (intensité OCT). En sortie = coordonnée axiale des pics correspondant à 
    #    (1) le couple couche de Bowman/membrane basale épithéliale 
    #    (2) l'endothélium (Descemet est hyporéflective juste avant)
    kpsh = False
    valley = False
#    idx = np.where(Mean_I_large > (np.amax(Mean_I_large)-np.mean(Mean_I_large[0:30]))/3)
#    idx = np.where(Mean_I_large > 1.5*np.quantile(Mean_I_large,0.75))
    idx = np.where(Mean_I_large > 1.5*np.mean(Mean_I_large))        
    ind = idx[0][0] + detect_peaks(Mean_I_large[idx], mph, mpd, threshold, edge, kpsh, valley, show=False)
    if aff_peaks == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, facecolor='w')
        ax.plot(Mean_I_large, linewidth=1)
        ax.plot(ind, Mean_I_large[(ind),], '+', mfc=None, mec='r', mew=2, ms=8)
        plt.title('Détection des pics du stroma : figure de contrôle')
        plt.xlabel('Profondeur [px]', fontsize = 15)
        plt.ylabel('Intensité OCT moyenne brute', fontsize = 15)
    if len(ind) < 3:
        detect_peaks(Mean_I_large, mph, mpd, threshold, edge, kpsh, valley, show=True)
#        raise ValueError('Pics non tous détectés : changez mph ou threshold')
#    print(ind)
#    coord_stroma = (np.zeros((2,1), dtype=int)) #np.concatenate
    coord_stroma = (ind[1], ind[-1])    # tuple = immutable
    coord_cornee = (ind[0], ind[-1])
    return coord_stroma, coord_cornee

def specular_cut_cursor(OCT, mean_signal, mean_signal_smooth, der1, pathname, pas_lat):
    # Définir à la main la zone de coupure centrale
    fig = plt.figure(2)
    plt.clf()
    ax = fig.add_subplot(111, facecolor='w')
    plt.imshow(OCT, cmap="gray")
    offset_aff = np.ceil(np.amax(mean_signal)) + 5
    #coeff = 20/(np.amax(der1)-np.amin(der1))
    lineN, = plt.plot(offset_aff + -mean_signal_smooth, color='tab:purple')
    lineO, = plt.plot(offset_aff + -10*der1, color='tab:green')
    ind = np.where(mean_signal_smooth/np.amax(mean_signal_smooth)>0.82)[0]  
    plt.vlines(ind[0], 0, OCT.shape[0], color='tab:purple', linestyle='dashed', linewidth=0.5)
    plt.vlines(ind[-1], 0, OCT.shape[0], color='tab:purple', linestyle='dashed', linewidth=0.5)    
    plt.title('Sélectionnez les bornes latérales du pic spéculaire')
    fig.cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='g', linewidth=1)
    R = plt.ginput(2)
    xmin_cut = int(min(round(R[0][0]), round(R[1][0])))
    xmax_cut = int(max(round(R[0][0]), round(R[1][0])))
#    OCT_cut = np.delete(OCT, range(xmin_cut, xmax_cut), axis=1)
    mean_signal = np.mean(OCT, axis = 0)
    coord_centre = xmin_cut + np.where(mean_signal[xmin_cut:xmax_cut] == np.amax(mean_signal[xmin_cut:xmax_cut]))[0][0]
    SpecularCut = int(round((xmax_cut-xmin_cut)*pas_lat))
    plt.clf()
    plt.imshow(OCT, cmap="gray")    
    lineW, = plt.plot(np.ones([len(OCT),1]), color='tab:green', linewidth=1.5, linestyle='dashed', label='Délimitation manuelle')
    plt.axvline(xmin_cut, ymin=0, ymax=OCT.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=1)
    plt.axvline(xmax_cut, ymin=0, ymax=OCT.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=1)
    plt.title("Coupure spéculaire manuelle : largeur = %i µm" %SpecularCut)
    plt.legend(fontsize=11, loc = 'lower right') 
    lineW.set_visible(False)
    plt.show()
        
    return xmin_cut, xmax_cut, coord_centre, SpecularCut

def cursor_segm_stroma(MsgBox, FlattenedImage2, marge_postBowman, marge_preDescemet, SpecularCut, colonne_inf, colonne_sup, coord_cornee, ep_stroma_tot_um, ep_stroma_analyse_um, pas, pas_lat, save, pathname, champ_lat, coord_centre2):
    # Segmentation manuelle du stroma sur image OCT aplatie
#    global result
#    if colonne_inf1 in locals():
#        del(colonne_inf1, colonne_sup1, coord_stroma1, profondeur_min1, profondeur_max1, profondeur_min1_um, profondeur_max1_um) # tuple
    fig = plt.figure(6)
    plt.clf()
    ax = fig.add_subplot(111, facecolor='w')
    ax.imshow(FlattenedImage2, cmap="gray")
    y2, y1 = ax.get_ylim() 
#    ax_twin = ax.twiny()
#    width_twin = (y2+y1)*pas_lat/1000
#    offset_width = 0.06*width_twin
#    ax_twin.hlines(0,xmin=-offset_width,xmax=width_twin+offset_width)
    plt.title('Sélectionnez les bornes latérales du stroma. Pointillés : champ de %.1f mm' %np.round(((colonne_sup-colonne_inf)*pas_lat+SpecularCut)/1000,1))
    plt.vlines(colonne_inf, ymin=y1, ymax=y2, color='r', linestyle='dashed', linewidth=1)
    plt.vlines(colonne_sup, ymin=y1, ymax=y2, color='r', linestyle='dashed', linewidth=1)
    fig.cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='r', linewidth=1)
    R1 = plt.ginput(2)    
    colonne_inf1 = int(min(round(R1[0][0]), round(R1[1][0])))
    colonne_sup1 = int(max(round(R1[0][0]), round(R1[1][0])))
    plt.close()
    
    if MsgBox == "no": # Sur l'image OCT
        fig = plt.figure(6)
        plt.clf()
        ax = fig.add_subplot(111, facecolor='w')
        plt.imshow(FlattenedImage2, cmap="gray")
        plt.axvline(colonne_inf1, color='r', linewidth=1, linestyle='dashed')
        plt.axvline(colonne_sup1, color='r', linewidth=1, linestyle='dashed')     
        plt.title('Délimitez la profondeur du stroma (Bowman/endothélium)')
        plt.xlabel('Profondeur [px]', fontsize=15)
        plt.ylabel('Intensité OCT moyenne', fontsize=15)        
        fig.cursor = Cursor(ax, horizOn=True, vertOn=False, useblit=True, color='r', linewidth=1)
        R2 = plt.ginput(2)
        coord_stroma1 = (int(min(round(R2[0][1]), round(R2[1][1]))),int(max(round(R2[0][1]), round(R2[1][1]))))
        profondeur_min1 = int(coord_stroma1[0] + marge_postBowman/pas)
        profondeur_max1 = int(coord_stroma1[1] - marge_preDescemet/pas)
        profondeur_min1_um = int(profondeur_min1*pas + marge_postBowman)
        profondeur_max1_um = int(profondeur_max1*pas - marge_preDescemet)
        
        
    else:# Sur le profil d'atténuation
#        Mean_I = np.mean(FlattenedImage2, axis=1)
#        N_pix_depth = np.shape(Mean_I)[0] # Profil d'atténuation
#        z = np.arange(0, N_pix_depth*pas, pas)
        profondeur_inf1 = 0
        profondeur_sup1 = coord_cornee[1] + 100
        CutImage = FlattenedImage2[profondeur_inf1:profondeur_sup1, colonne_inf1:colonne_sup1]
        Mean_I = np.mean(CutImage, axis=1)
        z = np.arange(0, np.shape(Mean_I)[0]*pas, pas)
        fig = plt.figure(6)
        ax = fig.add_subplot(111, facecolor='w')
#        idx_stop = np.where(Mean_I==np.max(Mean_I))[0][0] + np.shape(Mean_I)[0] #- np.where(np.flip(Mean_I[10:]) < 3)[0][-1] + 70
        plt.plot(Mean_I, linewidth=2)
        plt.title('Sélectionnez les pics de Bowmann et de l\'endothélium')
        plt.show()
        fig.cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='r', linewidth=1)
        R2 = plt.ginput(2)
#        profondeur_min = int(min(round(R2[0][0]), round(R2[1][0])))  # en pixels
#        profondeur_max = int(max(round(R2[0][0]), round(R2[1][0])))
        coord_stroma1 = (int(min(round(R2[0][0]), round(R2[1][0]))),int(max(round(R2[0][0]), round(R2[1][0]))))
        profondeur_min1 = int(coord_stroma1[0] + marge_postBowman/pas)
        profondeur_max1 = int(coord_stroma1[1] - marge_preDescemet/pas)
        profondeur_min1_um = int(profondeur_min1*pas + marge_postBowman)
        profondeur_max1_um = int(profondeur_max1*pas - marge_preDescemet)
#        plt.clf()
#        plt.plot(z, Mean_I, linewidth=2)
#        plt.axvline(profondeur_min_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
#        plt.axvline(profondeur_max_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
#        plt.xlabel('Profondeur [µm]')
#        plt.ylabel('Intensité OCT moyenne')
#        if save == 1:
#            plt.savefig(pathname + '/depth_profile_segmentation.png', dpi=200, bbox_inches='tight')
   
#        plt.close()
    ep_cornee_um = (coord_stroma1[1]-coord_cornee[0])*pas
    
        #    Affichage ###############################################################    
    CutImage = FlattenedImage2[:coord_stroma1[1]+100, colonne_inf1:colonne_sup1]
    Mean_I_large_adjust = np.mean(CutImage, axis=1)
    SD_I_large_adjust = np.std(CutImage, axis=1)#/np.sqrt(np.shape(CutImage)[1]-1)
    
    #% Tracé de la fonction profil avec segmentation
    N_pix_depth = np.shape(Mean_I_large_adjust)[0]
    z = np.arange(0, N_pix_depth, 1)
    z = z*pas
    
    fig = plt.figure(5)
    fig.clf()
    ax = fig.add_subplot(211, facecolor='w')
    ax.plot(z,Mean_I_large_adjust, linewidth=2)
    ax.plot(tuple(pas*np.array(coord_stroma1)), Mean_I_large_adjust[(coord_stroma1),], '+', mfc=None, mec='r', mew=2, ms=8)
    # attention virgule indispensable après tuple pour l'indexation de Mean_I_large
    plt.fill_between(z, Mean_I_large_adjust-SD_I_large_adjust, Mean_I_large_adjust+SD_I_large_adjust,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
    linewidth=0)
    plt.axvline(profondeur_min1_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(profondeur_max1_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)  
    plt.title('Segmentation du stroma en profondeur')
    text_str = 'Épaisseur totale cornée = %i µm\n Épaisseur totale stroma = %i µm (%.1f%%)\n Épaisseur stroma analysée = %i µm (%.1f%%)\n Champ analysé = %.1f mm' %(ep_cornee_um,ep_stroma_tot_um,100*round(ep_stroma_tot_um/ep_cornee_um,3),ep_stroma_analyse_um,100*round(ep_stroma_analyse_um/ep_cornee_um,3),np.round(((colonne_sup-colonne_inf)*pas_lat+SpecularCut)/1000,1))
    (ymin,ymax) = ax.get_ylim()
    plt.text(np.amax(z)*1.04,0.7*ymax,text_str,horizontalalignment='right',fontsize=9)
    plt.title('Segmentation du stroma')
    plt.xlabel('Profondeur [µm]', fontsize=15)

#    text_str = 'Épaisseur totale cornée = %i µm\n Épaisseur totale stroma = %i µm (%.1f%%)\n Épaisseur stroma analysée = %i µm (%.1f%%)' %(ep_cornee_um,ep_stroma_tot_um,100*round(ep_stroma_tot_um/ep_cornee_um,3),ep_stroma_analyse_um,100*round(ep_stroma_analyse_um/ep_cornee_um,3))
#    (ymin,ymax) = ax.get_ylim()
#    plt.text(np.amax(z)*0.94,-280,text_str,horizontalalignment='right',fontsize=9)
        
    ax = fig.add_subplot(212)
    plt.imshow(FlattenedImage2[:profondeur_max1+100,], cmap="gray")
    plt.hlines(profondeur_min1, xmin=colonne_inf1, xmax=colonne_sup1, color='r', linestyle='dashed', linewidth=1)
    plt.hlines(profondeur_max1, xmin=colonne_inf1, xmax=colonne_sup1, color='r', linestyle='dashed', linewidth=1)
    plt.vlines(colonne_inf1, ymin=profondeur_min1, ymax=profondeur_max1, color='r', linestyle='dashed', linewidth=1)
    plt.vlines(colonne_sup1, ymin=profondeur_min1, ymax=profondeur_max1, color='r', linestyle='dashed', linewidth=1)
    plt.axis('off')
    plt.show()
#    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#    plt.savefig(pathname + '/ROI.png', dpi=100, bbox_inches=extent)
    if save == True:
        plt.savefig(pathname + '/2_segmentation_stroma_champ_%.1fmm_std.png'%((colonne_sup-colonne_inf)*pas_lat/1000), dpi=200, bbox_inches='tight')
    
#        fig = plt.figure(6)
#        fig.clf()
#        ax = fig.add_subplot(111, facecolor='w')
#        ax.plot(z, Mean_I_large_adjust, linewidth=2)
#        ax.plot(tuple(pas*np.array(coord_stroma1)), Mean_I_large_adjust[(coord_stroma1),], '+', mfc=None, mec='r', mew=2, ms=8)
#        # attention virgule indispensable après tuple pour l'indexation de Mean_I_large
#        plt.axvline(profondeur_min1_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
#        plt.axvline(profondeur_max1_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
#        plt.title('Segmentation axiale du stroma')
#        plt.xlabel('Profondeur [µm]', fontsize=15)
#        plt.ylabel('Intensité OCT moyenne', fontsize=15)
#        plt.savefig(pathname + '/depth_profile_segmentation.png', dpi=200, bbox_inches='tight')
#        plt.close()
        
    # ROI sur image OCT    
#    if save == True:
#        fig = plt.figure(7)
#        plt.clf()
#        plt.imshow(FlattenedImage2, cmap="gray")
#        ax = fig.add_subplot(1, 1, 1)
#        plt.hlines(profondeur_min1, xmin=colonne_inf1, xmax=colonne_sup1, color='r', linestyle='dashed', linewidth=1)
#        plt.hlines(profondeur_max1, xmin=colonne_inf1, xmax=colonne_sup1, color='r', linestyle='dashed', linewidth=1)
#        plt.vlines(colonne_inf1, ymin=profondeur_min1, ymax=profondeur_max1, color='r', linestyle='dashed', linewidth=1)
#        plt.vlines(colonne_sup1, ymin=profondeur_min1, ymax=profondeur_max1, color='r', linestyle='dashed', linewidth=1)
#        plt.axis('off')
#        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        plt.savefig(pathname + '/ROI.png', dpi=100, bbox_inches=extent)
#        plt.close()
    
    if MsgBox == "no":
        fig = plt.figure(5)
        plt.show()
        
    return (colonne_inf1, colonne_sup1, coord_stroma1, profondeur_min1, profondeur_max1, profondeur_min1_um, profondeur_max1_um) # tuple

def specular_artefact_correction():
    # Calcule un masque de correction de l'hyperréflexion spéculaire du stroma 
    # postérieur, via deux ACP successives.
    
    
    
    return


#def specular_cut(OCT, VectNorm, pas_lat, quantile=80, largeur=0.5, show=False):
#    # !!!!!!!!!!! UTILISE VECTNORM AVANT DE L'AVOIR CRÉÉ !!!!!!!!!!!!!!!!!!!!!!
#    # Coupe la zone spéculaire d'une image OCT de cornée
#    # **largeur de coupe et champ latéral en [mm]
#    width = int(np.ceil(1000*largeur/pas_lat))   
#    # % Détermine l'intervalle correspondant à un quartile donnée
#    max_range = np.nonzero(VectNorm > np.percentile(VectNorm, quantile))
##    idx = np.transpose(np.asarray(max_range))
#    # % Prend sa valeur médiane
#    med_coord = int(np.ceil(np.median(max_range)))
#    xmin_cut = int(med_coord-width/2)
#    xmax_cut = int(med_coord+width/2)
#    OCT_cut = np.delete(OCT, range(xmin_cut, xmax_cut), axis=1)
#    if show == True:
#        print("med_coord = %i \nxmin_cut = %i \nxmax_cut = %i" %(med_coord, xmin_cut, xmax_cut))
#        #% Plot OCT avec délimitations zone spéculaire
#        plt.figure()
#        plt.imshow(OCT, cmap="gray")
#        plt.vlines(xmin_cut, ymin=0, ymax=OCT.shape[0], color='r', linestyle='dashed', linewidth=1)
#        plt.vlines(xmax_cut, ymin=0, ymax=OCT.shape[0], color='r', linestyle='dashed', linewidth=1)
#        plt.title('Délimitation de la zone spéculaire')        
#        #% Plot VectNorm
#        plt.figure()
#        lineN, = plt.plot(VectNorm, linewidth=2)
#        plt.title('Délimitation de la zone spéculaire') 
#        plt.xlabel('Pixels latéraux', fontsize = 15)
#        plt.ylabel('Intensité OCT moyenne [8-bit]', fontsize = 15)
#        plt.vlines(xmin_cut, ymin=min(VectNorm), ymax=265, color='r', linestyle='dashed', linewidth=1)
#        plt.vlines(xmax_cut, ymin=min(VectNorm), ymax=265, color='r', linestyle='dashed', linewidth=1)
#        plt.vlines(med_coord, ymin=min(VectNorm), ymax=265, color='r', linewidth=2)
##        ax = fig.add_subplot(111, facecolor='w')
##        plt.legend((lineN,lineM),('Délimitations zone spéculaire','Centre de la cornée imagée (B-scan)'), fontsize=18)
#        #% Plot superposition de moitiés de VectNorm
#        plt.figure()
#        lineN, =plt.plot(VectNorm[med_coord:])
#        lineM, =plt.plot(np.flip(VectNorm[:med_coord]))
#        plt.legend((lineM, lineN),('Partie droite','Partie gauche (flipped)'), fontsize=18)   
#        plt.title('Superposition des signaux de chaque côté du centre')
#        plt.xlabel('Distance au centre [px]', fontsize = 22)   
#        plt.ylabel('Intensité OCT moyenne [8-bit]', fontsize = 22)
#        
#    return OCT_cut, xmin_cut, xmax_cut