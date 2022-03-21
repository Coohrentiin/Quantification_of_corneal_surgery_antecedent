# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:12:07 2022

@author: Maëlle
Return the preprocessed SD-OCT corneal images (flattened + corrected for 
artifacts) acquired with RTVue-XR Avanti SD-OCT device by Optovue Inc.
"""

#%% USER INPUT

import os
from preproc_functions import preprocessing_OCT
# pathname = 'C:\\Users\\Maëlle\\Documents\\Rédaction\\_Article in vivo\\code' # chemin du dossier où est stockée l'image
dirname = os.getcwd()
# pathname=os.path.join(dirname,"..", 'Data','VILBERT_Maelle')
# filename = '_VILBERT_Maëlle__98276_Cornea Line_OD_2021-06-28_17-50-39_F_1994-02-07_1_sans-lumière-rouge-macro.jpg' # nom de l'image
pathname=os.path.join(dirname,"..", 'Data','HAZE')
filename = 'haze 1.jpg' # nom de l'image


champ_analyse_mm = 6        # [mm] width of analyzed area (comment the line if analyzing the full image width)
marge_postBowman = 60       # [µm] post-Bowman layer z-margin [default 60 µm]
marge_preDescemet = 30      # [µm] pre-Descemet membrane z-margin [default 30 µm]

save = False                # save figures and .txt/.csv files (bool)  
corr = True                 # compute posterior artefact correction (bool)
plots= False
#show = False               # display additional figures (bool)
user_validation = False     # ask user to validate segmentation results (bool)

#%% Preprocessing
from preproc_functions_21fev import preprocessing_OCT
import matplotlib.pyplot as plt

ProcessedImage, mask, coord_ROI, time_exe = preprocessing_OCT(filename, pathname, champ_analyse_mm, marge_postBowman, marge_preDescemet, save, corr, user_validation,plots)

"""
OUTPUT :
    ProcessedImage -- image aplatie et corrigées pour les artefacts
    mask -- masque de correction pour l'artefact du stroma postérieur, dans la zone correspondant à la zone d'intérêt (ROI) analysée
    coord_ROI -- ((limites en x),(limites en z)) du stroma analysé sur l'image ProcessedImage
    time_exe -- temps d'exécution
"""

if (0):
    plt.close('all')

# %%
plt.imshow(ProcessedImage)
plt.show()
# %%
