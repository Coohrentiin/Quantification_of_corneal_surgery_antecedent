from src.image_OCT import *
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

def gaus(x,a,x0,sigma):
    	return a*np.exp(-(x-x0)**2/(2*sigma**2))

def exp_decr(x, a, b, c):
    return a * np.exp(-b * x) + c

def create_new_dataset(repository,healthy_value,movingWin=False,eliminate=0,useExpCorr=False):
	files = [f for f in listdir(repository) if isfile(join(repository, f))]
	quantification=[]
	i=0
	i_list=[]
	for f in files:
		if(f[-3:]=="jpg"):
			i+=1
			try:
				im=image_OCT(os.path.join(repository, f))
			except:
				print("Error with: "+f+" image #"+str(i))
			try:
				im.AutoTreatment()
				im_Q=image_Quantification_3(im,movingWin=movingWin,eliminate=eliminate,useExpCorr=useExpCorr)
				# im_Q=image_Quantification_2(im,plot=False,intensity_corr=intensity_corr)
				parameters=im_Q.parameters
				parameters["Healthy"]=healthy_value
				parameters["Origin"]=f[:-4]
				quantification.append(parameters)
				# print(i,":",healthy)
				i_list.append(i)
			except:
				print("Error with: "+f+" image #"+str(i))
	n=len(quantification)
	print(n,"/",len(files)," files succesfully quantified")
	data=pd.DataFrame(quantification,index=range(n)) 
	return(data)

class image_Quantification_3(object):
	def __init__(self, image_OCT_element,movingWin=False,eliminate=0,useExpCorr=False):
		"""[Initialise the creation of the object image_Quantification by computiong the profile and calculate the quantifiers via Profile_quantification(_moving) and Quantification_parameters(_moving)]
		Args:
			image_OCT_element ([image_OCT]): [object of the class image_OCT]
			plot (bool, optional): [to rather plot fligure automatically or not]. Defaults to False.
			moving (bool, optional): [use the moving widow algorithm (True) or on the entire image (False)]. Defaults to False.

		Raises:
			ValueError: [Error during profile quantification or reation of the parameters]
		"""
		self.im=image_OCT_element
		if eliminate==0:
			self.image=self.im.OCT_flat
		else:
			width=self.im.OCT_flat.shape[1]
			to_eliminate=int(width*eliminate/100/2)
			self.image=self.im.OCT_flat[:,to_eliminate:(width-to_eliminate)]
		try:
			window=100
			N=self.im.OCT_flat.shape[1]-window
			N_p=N
			if movingWin:
				PeakWidth=0;Sigma=0;DataCov=0;Mean=0;MSE=0;Area_ratio=0;Alpha=0;Beta=0;IntensityPeak=0
				for i in range(N):
					self.image=self.im.OCT_flat[:,i:(i+window)]
					# print(self.image)
					try:
						self.Profile_quantification()
						self.Quantification_parameters(useExpCorr)
						PeakWidth+=self.PeakWidth
						Sigma+=self.Sigma
						DataCov+=self.DataCov
						Mean+=self.Mean
						MSE+=self.MSE
						Area_ratio+=self.Area_ratio
						Alpha+=self.Alpha
						Beta+=self.Beta
						IntensityPeak+=self.IntensityPeak
					except:
						N_p-=1
				self.PeakWidth=PeakWidth/N_p
				self.Sigma=Sigma/N_p
				self.DataCov=DataCov/N_p
				self.Mean=Mean/N_p
				self.MSE=MSE/N_p
				self.Area_ratio=Area_ratio/N_p
				self.Alpha=Alpha/N_p
				self.Beta=Beta/N_p
				self.IntensityPeak=IntensityPeak/N_p
				self.parameters={
					"PeakWidth": self.PeakWidth,
					"Sigma": self.Sigma,
					"DataCov": self.DataCov,
					"Mean": self.Mean,
					"MSE": self.MSE,
					"AreaRatio": self.Area_ratio,
					"Alpha": self.Alpha,
					"Beta": self.Beta,
					"IntensityPeak":self.IntensityPeak
				}
				# print(str(N_p)+"/"+str(N)+" window computed")
			else:
				self.Profile_quantification()
				self.Quantification_parameters(useExpCorr)
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification init: Image quantification impossible"
			raise ValueError(message)

	def getPeaks(self,intentityProfile,displayedPeak=3):
		"""[Return the #displayedPeak first peak position on the intencity profile in a 1D np-array]

		Args:
			intentityProfile ([1D np-array]): [Mean axial profile]
			displayedPeak (int, optional): [# of peaks to detect]. Defaults to 3.
		"""
		indexMax=np.argmax(intentityProfile)
		peak=argrelextrema(intentityProfile, np.greater)
		npeak1=np.unique(np.where(peak >= indexMax, peak,indexMax)[0])
		return(npeak1[0:displayedPeak])


	def getLowPeak(self,profile,peaks):
		"""[From the position of the second peak of the intensity profile, find the first local hollows in both direction arround the peak.
  			Hollows are detected by looking for local minimum, in case there is no local min:
     		- If it is a constant decrease, then lowpeak is define as the last ascan position available on the side studied
       		A filter is then applied:
         	The two lowpeaks are selected to have the same distance from the central peak as the minimal distance from the central peak to the two possition of lateral hollows]

		Args:
			peak ([1D np-array]): [first peaks position on the axial intencity profile]
			cropedProfile ([type]): [description]
			xmin ([type]): [description]
		"""
		peakLow=argrelextrema(profile, np.less)+self.xmin
		low1=np.unique(np.where(peakLow < peaks[1], peakLow,peaks[1])[0])#[:-1]
		if(low1.shape[0]>1):
				xlow1=low1[-1]
				if xlow1==peaks[1]:
					xlow1=low1[-2]
		else:
			xlow1=low1[0]
		
		low2=np.where(peakLow >= peaks[1], peakLow,peaks[1])[0]
		low2=low2[profile[low2-self.xmin]<0.8*profile[peaks[1]-self.xmin]]
		low2=np.unique(low2)
		if(low2.shape[0]>1):
			xlow2=low2[0]
			if xlow2==peaks[1]:
				xlow2=low2[1]
		elif (low2.shape[0]==1):
			xlow2=low2[0]
			if xlow2==peaks[1]:
				xlow2=profile.shape[0]+self.xmin-1
		else:
			xlow2=profile.shape[0]+self.xmin-1
		self.hollows=[xlow1,xlow2]

	def Profile_quantification(self,displayedPeak=3,pre_window=2,post_window=100):
		"""[summary]
		Args:
			displayedPeak (int, optional): [#of peak to compute]. Defaults to 2.
			window (int, optional): [number of ascan to consider left to first peak and right to second peak for ploting]. Defaults to 10.

		Raises:
			ValueError: [Error if the profile can be averaged]
		"""
		self.pre_window=pre_window
		self.post_window=post_window
		try:
			self.intentityProfile=np.mean(self.image,1)							#Compute the profile
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification: OCT_flat do not exist, run AutoTreatment or Flatenning before Profile_quantification"
			raise ValueError(message)

		peak=self.getPeaks(self.intentityProfile,displayedPeak)
		xdata=np.arange(0,post_window)
		ydata=self.intentityProfile[peak[0]:peak[0]+post_window]
		ydata=(ydata-np.min(ydata))/(np.max(ydata)-np.min(ydata))
		popt, pcov = curve_fit(exp_decr, xdata, ydata)
		ydata_fit=exp_decr(xdata, *popt)
		self.iProfileCrop=ydata
		self.iProfileExp=ydata_fit
		self.iProfileExpCorrection=self.iProfileCrop-self.iProfileExp
		self.xmin=peak[0]
		self.Alpha=popt[0]
		self.Beta=popt[1]
		self.peaks=peak

	def Quantification_parameters(self,useExpCorr=False):
		intensity_peak_bowman=self.iProfileExpCorrection[self.peaks[1]-self.xmin]
		intensity_peak_under_bowman=self.iProfileExpCorrection[self.peaks[2]-self.xmin]
		self.underBowman=False
		if(intensity_peak_bowman<=intensity_peak_under_bowman/0.9):
			self.underBowman=True
			peaks=self.peaks[1:]
		else:
			peaks=self.peaks
		self.IntensityPeak=self.iProfileCrop[peaks[1]-self.xmin]
		if useExpCorr:
			self.getLowPeak(self.iProfileExpCorrection,peaks)
			self.iProfileBell=self.iProfileExpCorrection[(self.hollows[0]-self.xmin):(self.hollows[1]+1-self.xmin)]
		else:
			self.getLowPeak(self.iProfileCrop,peaks)
			self.iProfileBell=self.iProfileCrop[(self.hollows[0]-self.xmin):(self.hollows[1]+1-self.xmin)]
		self.Study_bell_curve(useExpCorr)
		self.parameters={
			"PeakWidth": self.PeakWidth,
			"Sigma": self.Sigma,
			"DataCov": self.DataCov,
			"Mean": self.Mean,
			"MSE": self.MSE,
			"AreaRatio": self.Area_ratio,
			"Alpha": self.Alpha,
			"Beta": self.Beta,
			"IntensityPeak":self.IntensityPeak
		}
		return(0)

	def Study_bell_curve(self,useExpCorr=False):
		PeakWidth=(self.hollows[1]-self.hollows[0])*self.im.pas
		ydata_init=self.iProfileBell-np.min(self.iProfileBell)
		n=ydata_init.shape[0]
		xdata_init=np.arange(0,n)
		mean=np.argmax(ydata_init)
		sigma=0.1
		tck = interpolate.splrep(xdata_init, ydata_init, s=0)
		xdata = np.arange(0, n-1, 1/5) 				#Multiply by 5 points number
		ydata = interpolate.splev(xdata, tck, der=0)
		popt,pcov = curve_fit(gaus,xdata,ydata,p0=[1,mean,sigma])
		## Std of the gaussian function
		sigma=popt[2]*self.im.pas
		## Mean position of gaussian function
		mean=(popt[1]+self.hollows[0]-self.xmin)*self.im.pas
		ydata_fit=gaus(xdata_init,*popt)
		## Correlation between fitted data and initial data
		DataCov=np.cov(ydata_init,ydata_fit)
		DataCov=DataCov[0,1]
		## MSE between fitted data and initial data
		MSE=np.sqrt(np.sum((ydata_init-ydata_fit)**2))
  
		# Area calculation
		if useExpCorr:
			ydata=self.iProfileExpCorrection-np.min(self.iProfileExpCorrection)
		else:
			ydata=self.iProfileCrop-np.min(self.iProfileCrop)
		xdata=np.arange(0,len(ydata))
		x1=self.hollows[0];y1=ydata[self.hollows[0]-self.xmin]
		x2=self.hollows[1];y2=ydata[self.hollows[1]-self.xmin]
		m=(y1-y2)/(x1-x2);p=y1-m*x1
		x_lin_interp=np.arange(x1,x2,1)
		y_lin_interp=m*x_lin_interp+p
		ydata_c=np.concatenate((ydata[:(self.hollows[0]-self.xmin)],y_lin_interp,ydata[(self.hollows[1]-self.xmin):]))
		area_exp = np.trapz(ydata_c, dx=1)
		area_tot = np.trapz(ydata, dx=1)
		area_ratio = (area_tot-area_exp)/area_tot  
  
		self.iProfileBellGauss=ydata_fit
		self.MSE=MSE
		self.Sigma=sigma
		self.Mean=mean
		self.PeakWidth=PeakWidth
		self.DataCov=DataCov
		self.Area_ratio=area_ratio
  
	def Plot_ProfileExp(self):
		try:
			fig,ax=plt.subplots(1,3,figsize=(15,5))
			ax[0].imshow((self.image[self.peaks[0]-self.pre_window:self.peaks[0]+self.post_window][:]).T,aspect="auto",cmap="gray")
			ax[1].plot(self.iProfileCrop,'b')
			ax[1].plot(self.iProfileExp,'r--',label="a={:.2}, b={:.2}".format(self.Alpha,self.Beta))
			ax[1].plot(self.peaks-self.xmin,self.iProfileCrop[self.peaks-self.xmin],"bo",label="Peaks")
			ax[1].legend()
			ax[1].set_xlim([0,self.pre_window+self.post_window])
			ax[1].set_ylim([0,1])
			ax[2].plot(self.iProfileExpCorrection,'orange',label='Corrected profile')
			ax[2].plot(self.peaks-self.xmin,self.iProfileExpCorrection[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
			# ax[2].plot(savgol_filter(self.iProfileExpCorrection, 5, 2),'--')
			ax[2].legend()
			plt.show()
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification/Plot_ProfileExp: run Profile_quantification and Quantification_parameters before"
			raise ValueError(message)

	def Plot_ProfileQuantif(self,useExpCorr=False):
		try:
			fig,ax=plt.subplots(1,2,figsize=(15,5))
			if useExpCorr:
				ax[0].plot(self.iProfileExpCorrection,'orange',label='Corrected profile')
				ax[0].plot(self.peaks-self.xmin,self.iProfileExpCorrection[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
				ax[0].plot(self.hollows-self.xmin,self.iProfileExpCorrection[self.hollows-self.xmin],color="green",marker="o",linestyle='',label="Hollows")
			else:
				ax[0].plot(self.iProfileCrop,'orange',label='Corrected profile')
				ax[0].plot(self.peaks-self.xmin,self.iProfileCrop[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
				ax[0].plot(self.hollows-self.xmin,self.iProfileCrop[self.hollows-self.xmin],color="green",marker="o",linestyle='',label="Hollows")
			ax[0].set_title("Fibrosis under bowman detected:"+str(self.underBowman))
			ax[0].legend()
			ax[1].plot(self.iProfileBell-np.min(self.iProfileBell),label="Extracted bell profile")
			ax[1].plot(self.iProfileBellGauss,'r--',label="N(mean={:.2},sigma={:.2})".format(self.Mean,self.Sigma))
			ax[1].legend()
			plt.show()
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification/Plot_ProfileQuantif: run Profile_quantification before"
			raise ValueError(message)

	def Plot_All(self):
		try:
			fig,ax=plt.subplots(3,2,figsize=(10,15))
			ax[0][0].imshow((self.image[self.peaks[0]-self.pre_window:self.peaks[0]+self.post_window][:]).T,aspect="auto",cmap="gray")
			ax[1][0].plot(self.iProfileCrop,'b')
			ax[1][0].plot(self.iProfileExp,'r--',label="a={:.2}, b={:.2}".format(self.Alpha,self.Beta))
			ax[1][0].plot(self.peaks-self.xmin,self.iProfileCrop[self.peaks-self.xmin],"bo",label="Peaks")
			ax[1][0].legend()
			ax[1][0].set_xlim([0,self.pre_window+self.post_window])
			ax[1][0].set_ylim([0,1])
			ax[2][0].plot(self.iProfileExpCorrection,'orange',label='Corrected profile')
			ax[2][0].plot(self.peaks-self.xmin,self.iProfileExpCorrection[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
			# ax[2].plot(savgol_filter(self.iProfileExpCorrection, 5, 2),'--')
			ax[2][0].legend()
   
			ax[0][1].plot(self.iProfileExpCorrection,'orange',label='Corrected profile')
			ax[0][1].plot(self.peaks-self.xmin,self.iProfileExpCorrection[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
			ax[0][1].plot(self.hollows-self.xmin,self.iProfileExpCorrection[self.hollows-self.xmin],color="green",marker="o",linestyle='',label="Hollows")
			ax[0][1].set_title("Fibrosis under bowman detected:"+str(self.underBowman))
			ax[0][1].legend()
			ax[1][1].plot(self.iProfileBell-np.min(self.iProfileBell),label="Extracted bell profile")
			ax[1][1].plot(self.iProfileBellGauss,'r--',label="N(mean={:.2},sigma={:.2})".format(self.Mean,self.Sigma))
			ax[1][1].legend()
			plt.show()
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification/Plot_ProfileQuantif: run Profile_quantification before"
			raise ValueError(message)