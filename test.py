from image_OCT import *
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

def plot_treatment(im):
	plt.subplot(221)
	plt.title("Input image")
	plt.imshow(im.OCT_brut,cmap="gray")
	plt.axis("off")
	plt.subplot(222)
	plt.title("Arrow removal")
	plt.imshow(im.OCT,cmap="gray")
	plt.axis("off")
	plt.subplot(223)
	plt.title("Exposure correction")
	plt.imshow(im.OCT_cut,cmap="gray")
	plt.axis("off")
	plt.subplot(224)
	plt.title("Flattening")
	plt.imshow(im.OCT_flat,cmap="gray")
	plt.axis("off")
	plt.show()

def plot_profile(im,res):
	peak=res[0];xmin=res[1];xmax=res[2];xlow1=res[3];xlow2=res[4];extractedProf=res[5]	
	plt.subplot(211)
	plt.imshow(im.OCT_flat.transpose(),cmap="gray",aspect="auto")
	plt.axis("off")
	plt.subplot(212)
	plt.plot(im.intentityProfile)
	plt.xlim([0,im.intentityProfile.shape[0]])
	plt.plot(peak,im.intentityProfile[peak],'+')
	plt.show()

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def quantify(res,plot=False):
	peak=res[0];xmin=res[1];xmax=res[2];xlow1=res[3];xlow2=res[4];extractedProf=res[5]
	PeakWidth=xlow2-xlow1
	#
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
	sigma=popt[2]
	mean=popt[1]
	y_new_fit=gaus(xnew,*popt)
	DataCov=np.cov(ynew,y_new_fit)
	DataCov=DataCov[0,1]
	MSE=np.sqrt(np.sum((ynew-y_new_fit)**2))
	if plot:
		plt.plot(xnew,ynew,label='Data interpolation')
		plt.plot(x,y,'b+:',label='Data')
		plt.plot(xnew,y_new_fit,'ro:',label='fit $sigma$={:.2f}, cov={:.3f}, MSE={:.3f}'.format(sigma,DataCov,MSE))
		plt.legend()
		plt.show()
	return(PeakWidth,sigma,DataCov,mean,MSE)

filepathname="C:/Users/csoub/OneDrive/Bureau/3A/Ensta/Super projet/Algo_Maelle/Data/VILBERT_Maelle/test.jpg"
im=image_OCT(filepathname)
im.AutoTreatment()
res=im.Profile_quantification()
quantification=quantify(res,True)
# plot_treatment(im)
# plot_profile(im,res)
print(quantification)

filepathname="C:/Users/csoub/OneDrive/Bureau/3A/Ensta/Super projet/Algo_Maelle/Data/HAZE/haze 1.jpg"
im=image_OCT(filepathname)
im.AutoTreatment()
res=im.Profile_quantification()
quantification=quantify(res,True)
# plot_treatment(im)
# plot_profile(im,res)
print(quantification)