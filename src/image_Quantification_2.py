import numpy as np
import scipy.ndimage as img
import scipy.signal as sgn
from copy import deepcopy
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import interpolate
from matplotlib import pyplot as plt

def gaus(x,a,x0,sigma):
    	return a*np.exp(-(x-x0)**2/(2*sigma**2))

def exp_decr(x,k,a,b,l):
	return(a*exp(-l*(x-k))+b)

class image_Quantification_2(object):
	def __init__(self, image_OCT_element,plot=False,intensity_corr=False):
		"""[Initialise the creation of the object image_Quantification by computiong the profile and calculate the quantifiers via Profile_quantification(_moving) and Quantification_parameters(_moving)]

		Args:
			image_OCT_element ([image_OCT]): [object of the class image_OCT]
			plot (bool, optional): [to rather plot fligure automatically or not]. Defaults to False.
		Raises:
			ValueError: [Error during profile quantification or reation of the parameters]
		"""
		self.im=image_OCT_element
		try:
			# self.res=self.Profile_quantification()
			self.res=self.Profile_quantification()
			self.parameters=self.Quantification_parameters(plot,intensity_corr)
			# self.param=param
			# self.PeakWidth=param[0]
			# self.sigma=param[1]
			# self.DataCov=param[2]
			# self.mean=param[3]
			# self.MSE=param[4]
			# self.area_ratio=param[5]
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification init: Image quantification impossible"
			raise ValueError(message)

	def getPeaks(self,intentityProfile,displayedPeak=2):
		"""[Return the #displayedPeak first peak position on the intencity profile in a 1D np-array]

		Args:
			intentityProfile ([1D np-array]): [Mean axial profile]
			displayedPeak (int, optional): [# of peaks to detect]. Defaults to 2.
		"""
		indexMax=np.argmax(intentityProfile)
		peak=argrelextrema(intentityProfile, np.greater)
		npeak1=np.unique(np.where(peak >= indexMax, peak,indexMax)[0])
		return(npeak1[0:displayedPeak])

	def getLowPeak(self,peak,cropedProfile,xmin):
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
		peakLow=argrelextrema(cropedProfile, np.less)+xmin
		low1=np.unique(np.where(peakLow < peak[1], peakLow,peak[1])[0])#[:-1]
		if(low1.shape[0]>1):
				xlow1=low1[-1]
				if xlow1==peak[1]:
					xlow1=low1[-2]
		else:
			xlow1=low1[0]
		
		# low2=np.unique(np.where(peakLow >= peak[1], peakLow,peak[1])[0])#[1:]
		low2=np.where(peakLow >= peak[1], peakLow,peak[1])[0]
		low2=np.unique(np.where(cropedProfile[low2-xmin] <= cropedProfile[xlow1-xmin], peakLow,peak[1])[0])
		if(low2.shape[0]>1):
			xlow2=low2[0]
			if xlow2==peak[1]:
				xlow2=low2[1]
		elif (low2.shape[0]==1):
			xlow2=low2[0]
			if xlow2==peak[1]:
				xlow2=cropedProfile.shape[0]+xmin-1
		else:
			xlow2=cropedProfile.shape[0]+xmin-1
		
		xlow1_total=xlow1
		xlow2_total=xlow2
		# dist1=peak[1]-xlow1
		# dist2=xlow2-peak[1]
		# # dist_avg=int((dist1+dist2)/2)
		# if dist1>dist2:
		# 	# xlow1=peak[1]-dist_agv
		# 	xlow1=peak[1]-dist2
		# else:
		# 	# xlow2=peak[1]+dist_avg
		# 	xlow2=peak[1]+dist1
		# if xlow1<0:
		# 	xlow1=0
		# if xlow2>cropedProfile.shape[0]+xmin-1:
		# 	xlow2=cropedProfile.shape[0]+xmin-1
		return(xlow1,xlow2,xlow1_total,xlow2_total)

	def Profile_quantification(self,displayedPeak=3,window=10):
		"""[summary]

		Args:
			displayedPeak (int, optional): [#of peak to compute]. Defaults to 2.
			window (int, optional): [number of ascan to consider left to first peak and right to second peak for ploting]. Defaults to 10.

		Raises:
			ValueError: [Error if the profile can be averaged]
		"""
		try:
			self.intentityProfile=np.mean(self.im.OCT_flat,1)							#Compute the profile
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification: OCT_flat do not exist, run AutoTreatment or Flatenning before Profile_quantification"
			raise ValueError(message)
		peak=self.getPeaks(self.intentityProfile,displayedPeak)
  
		intensity_peak_bowman=self.intentityProfile[peak[1]]
		intensity_peak_under_bowman=self.intentityProfile[peak[2]]
		# print(intensity_peak_bowman,intensity_peak_under_bowman)
		self.underBowman=False
		if(intensity_peak_bowman<=intensity_peak_under_bowman/0.9):
			# print(self.im.Path_patient,": potential fibrosis under Bowman")
			self.underBowman=True
			peak=peak[1:]
		xmin=peak[0]-window;xmax=peak[-1]+window										#Select profion of the profile between the two first peak plus a small window (for plot)
		cropedProfile=self.intentityProfile[xmin:xmax]
		xlow1,xlow2,xlow1_total,xlow2_total=self.getLowPeak(peak,cropedProfile,xmin) 	#Get the corrected hollow position
		extractedProfile=self.intentityProfile[xlow1:xlow2+1]							#Profile betwwen hollow for fitting
		self.extractedProfile=extractedProfile
		return(peak,xmin,xmax,xlow1,xlow2,extractedProfile,cropedProfile,xlow1_total,xlow2_total)


	def Quantification_parameters(self,plot=False,intensity_corr=False):
		peak=self.res[0];xmin=self.res[1];xmax=self.res[2];xlow1=self.res[3];xlow2=self.res[4];extractedProf=self.res[5];xlow1_total=self.res[7];xlow2_total=self.res[8];
		### Profile fitting for quantification: 
  		### ----------------------------------- 
		## Peak width converted in um
		PeakWidth=(xlow2-xlow1)/self.im.pas
		data=extractedProf
		if intensity_corr:
			y=data-( (data[-1]-data[0])/(len(data))*np.arange(0,len(data),1)+data[0] ) # With intensity correction
		else:		
  			y=data-np.min(data) # Without intensity correction 
		# print(y)
		n=y.shape[0]
		x=np.arange(0,n)
		mean=np.argmax(y)
		sigma=0.1
		tck = interpolate.splrep(x, y, s=0)
		xnew = np.arange(0, n-1, 1/5) 				#Multiply by 5 points number
		ynew = interpolate.splev(xnew, tck, der=0)
		popt,pcov = curve_fit(gaus,xnew,ynew,p0=[1,mean,sigma])
		## Std of the gaussian function
		sigma=popt[2]/self.im.pas
		## Mean position of gaussian function
		mean=popt[1]/self.im.pas
		y_new_fit=gaus(xnew,*popt)
		## Correlation between fitted data and initial data
		DataCov=np.cov(ynew,y_new_fit)
		DataCov=DataCov[0,1]
		## MSE between fitted data and initial data
		MSE=np.sqrt(np.sum((ynew-y_new_fit)**2))
	
		# Area calculation
		window=100
		x_x=np.arange(peak[0],peak[0]+window,1)
		subdata=self.intentityProfile[peak[0]:peak[0]+window]
		x1=xlow1_total;y1=self.intentityProfile[xlow1_total]
		x2=xlow2_total;y2=self.intentityProfile[xlow2_total]
		m=(y1-y2)/(x1-x2);p=y1-m*x1
		x_lin_interp=np.arange(x1,x2,1)
		y_lin_interp=m*x_lin_interp+p
		ydata_c=np.concatenate((self.intentityProfile[peak[0]:xlow1_total],y_lin_interp,self.intentityProfile[xlow2_total:peak[0]+window]))
		# ydata=np.concatenate((self.intentityProfile[peak[0]:xlow1_total],self.intentityProfile[xlow2_total:peak[0]+window]))
		# xdata=np.concatenate((x_x[:xlow1_total-peak[0]],x_x[xlow2_total-peak[0]:]))
		# tck = interpolate.splrep(xdata, ydata, s=0)
		# ydata_c = interpolate.splev(x_x, tck, der=0)
		area_exp = np.trapz(ydata_c, dx=1)
		area_tot = np.trapz(subdata, dx=1)
		area_ratio = (area_tot-area_exp)/area_tot

		if plot:
			plt.plot(xnew,ynew,label='Data interpolation')
			plt.plot(x,y,'b+:',label='Data')
			plt.plot(xnew,y_new_fit,'ro:',label='fit $sigma$={:.2f}, cov={:.3f}, MSE={:.3f}'.format(sigma,DataCov,MSE))
			plt.legend()
			plt.show()

			plt.figure()
			plt.plot(x_x,subdata,'b',label="initial data A={:.2f}".format(area_tot),linewidth=2)
			plt.plot(xlow1_total,self.intentityProfile[xlow1_total],'r+')
			plt.plot(xlow2_total,self.intentityProfile[xlow2_total],'r+')
			plt.plot(x_x,ydata_c,'r--',label="croped and interpolated data A={:.2f}".format(area_exp),linewidth=2)
			plt.title("Area ratio {:.2f}%".format(area_ratio*100))
			plt.legend()
			plt.show()
		parameters ={
			"PeakWidth": PeakWidth,
			"Sigma": sigma,
			"DataCov": DataCov,
			"Mean": mean,
			"MSE": MSE,
			"AreaRatio": area_ratio,
			"FibrosisUnderBowman": self.underBowman
		}
		return(parameters)