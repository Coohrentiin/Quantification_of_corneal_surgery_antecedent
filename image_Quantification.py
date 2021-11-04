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

class image_Quantification(object):
	def __init__(self, image_OCT_element,plot=False):
		self.im=image_OCT_element
		# print(type(image_OCT_element))
		# plt.imshow(image_OCT_element.OCT_flat.transpose(),cmap="gray",aspect="auto")
		# plt.show()
		try:
			self.res=self.Profile_quantification()
			param=self.Quantification_parameters(plot)
			self.param=param
			self.PeakWidth=param[0]
			self.sigma=param[1]
			self.DataCov=param[2]
			self.mean=param[3]
			self.MSE=param[4]
			self.area_ratio=param[5]
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification init: Image quantification impossible"
			raise ValueError(message)

	def getPeaks(self,intentityProfile,displayedPeak=2):
		indexMax=np.argmax(intentityProfile)
		peak=argrelextrema(intentityProfile, np.greater)
		npeak1=np.unique(np.where(peak >= indexMax, peak,indexMax)[0])
		return(npeak1[0:displayedPeak])

	def getLowPeak(self,peak,cropedProfile,xmin):
		peakLow=argrelextrema(cropedProfile, np.less)+xmin
		low1=np.unique(np.where(peakLow < peak[1], peakLow,peak[1])[0])#[:-1]
		if(low1.shape[0]>1):
				xlow1=low1[-1]
				if xlow1==peak[1]:
					xlow1=low1[-2]
		else:
			xlow1=low1[0]
		
		low2=np.unique(np.where(peakLow >= peak[1], peakLow,peak[1])[0])#[1:]
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
		dist1=peak[1]-xlow1
		dist2=xlow2-peak[1]
		# dist_avg=int((dist1+dist2)/2)
		if dist1>dist2:
			# xlow1=peak[1]-dist_agv
			xlow1=peak[1]-dist2
		else:
			# xlow2=peak[1]+dist_avg
			xlow2=peak[1]+dist1
		# if xlow1<0:
		# 	xlow1=0
		# if xlow2>cropedProfile.shape[0]+xmin-1:
		# 	xlow2=cropedProfile.shape[0]+xmin-1
		return(xlow1,xlow2,xlow1_total,xlow2_total)

	def Profile_quantification(self,displayedPeak=2,window=10):
		try:
			self.intentityProfile=np.mean(self.im.OCT_flat,1)
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification: OCT_flat do not exist, run AutoTreatment or Flatenning before Profile_quantification"
			raise ValueError(message)
		peak=self.getPeaks(self.intentityProfile,displayedPeak)

		xmin=peak[0]-window;xmax=peak[-1]+window
		cropedProfile=self.intentityProfile[xmin:xmax]
		xlow1,xlow2,xlow1_total,xlow2_total=self.getLowPeak(peak,cropedProfile,xmin)
		extractedProfile=self.intentityProfile[xlow1:xlow2+1]
		self.extractedProfile=extractedProfile
		extractedProf=self.intentityProfile[xlow1:xlow2+1]
		return(peak,xmin,xmax,xlow1,xlow2,extractedProf,cropedProfile,xlow1_total,xlow2_total)

	def Quantification_parameters(self,plot=False):
		peak=self.res[0];xmin=self.res[1];xmax=self.res[2];xlow1=self.res[3];xlow2=self.res[4];extractedProf=self.res[5];xlow1_total=self.res[7];xlow2_total=self.res[8];
		PeakWidth=(xlow2-xlow1)/self.im.pas
		# Profile fitting for quantification: 
		data=extractedProf
		y=data-np.min(data)
		n=y.shape[0]
		x=np.arange(0,n)
		mean=np.argmax(y)
		sigma=0.1
		tck = interpolate.splrep(x, y, s=0)
		xnew = np.arange(0, n-1, 1/5) 				#Multiply by 5 points number
		ynew = interpolate.splev(xnew, tck, der=0)
		popt,pcov = curve_fit(gaus,xnew,ynew,p0=[1,mean,sigma])
		sigma=popt[2]/self.im.pas
		mean=popt[1]/self.im.pas
		y_new_fit=gaus(xnew,*popt)
		DataCov=np.cov(ynew,y_new_fit)
		DataCov=DataCov[0,1]
		MSE=np.sqrt(np.sum((ynew-y_new_fit)**2))
	
				
		window=100
		x_x=np.arange(peak[0],peak[0]+window,1)
		subdata=self.intentityProfile[peak[0]:peak[0]+window]
		ydata=np.concatenate((self.intentityProfile[peak[0]:xlow1_total],self.intentityProfile[xlow2_total:peak[0]+window]))
		xdata=np.concatenate((x_x[:xlow1_total-peak[0]],x_x[xlow2_total-peak[0]:]))
		tck = interpolate.splrep(xdata, ydata, s=0)
		ydata_c = interpolate.splev(x_x, tck, der=0)
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
			plt.plot(x_x,subdata,'b',label="initial data A={:.2f}".format(area_tot))
			plt.plot(xlow1_total,self.intentityProfile[xlow1_total],'r+')
			plt.plot(xlow2_total,self.intentityProfile[xlow2_total],'r+')
			plt.plot(x_x,ydata_c,'r--',label="croped and interpolated data A={:.2f}".format(area_exp))
			plt.title("Area ratio {:.2f}%".format(area_ratio*100))
			plt.legend()
			plt.show()
		return(PeakWidth,sigma,DataCov,mean,MSE,area_ratio)
