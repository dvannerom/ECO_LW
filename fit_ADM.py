import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.integrate import quad
import math
import argparse
import psutil
import os
from netcdf_io import load_data, write_dataset

mpl.rcParams["mathtext.default"] = 'regular'

def elmer(x):
	cos = np.cos(np.radians(x))
	cos55 = np.cos(np.radians(55))
	return np.log(cos+2)/np.log(cos55+2)

def fit_fct_diff(x, a, b):
	term1 = a*(elmer(x[0])-1) + b*(elmer(x[0])**2-1)
	term2 = a*(elmer(x[1])-1) + b*(elmer(x[1])**2-1)
	return term1-term2

def fit_fct_diff_1D(x, a, b):
	return a*(elmer(x)-1) + b*(elmer(x)**2-1)

def radiance_lin(x, b):
	x = elmer(x)
	return 1 + b*(x-1)

def radiance_integrand(x, b):
	return radiance_lin(x, b)*np.sin(np.radians(x))*np.radians(1)

def radiance_lin_norm(x, a):
	b = x[1]
	return a*radiance_lin(x[0], b)

def radiance_lin_ratio(x, b):
	return radiance_lin(x[0], b)/radiance_lin(x[1], b)

def radiance_quad(x, b, c):
	x = elmer(x)
	return 1 + b*x + c*x**2

def radiance_quad_norm(x, a):
	b = x[1]
	c = x[2]
	return a*radiance_quad(x[0], b, c)

def radiance_quad_ratio(x, b, c):
	return radiance_quad(x[0], b, c)/radiance_quad(x[1], b, c)

def flux_integrand(x, a, b):
	return radiance_norm((x,b), a)*np.sin(np.radians(x))

def flux_upward_integrand(x, a, b):
	return radiance_norm((x,b), a)*np.cos(np.radians(x))*np.sin(np.radians(x))

if __name__ == '__main__':
    
	parser = argparse.ArgumentParser(description="sample argument parser")
	parser.add_argument("-d","--day", type=int, default=245)
	parser.add_argument("-r","--resolution", type=int, default=2)
	parser.add_argument("--no-plot", action="store_true", help="Skip plotting during fit; generate plots offline with plot_fit_ADM.py")
	#parser.add_argument("-c","--channel", type=int, default=0)
	#parser.add_argument("-s","--scene", type=int, default=0)
	args = parser.parse_args()

	day = args.day
	res = args.resolution	
	#channel = args.channel
	#scene = args.scene

	data_file = "data/preprocessed_files/abi_"+str(day)+"_res"+str(res)+"km_step1.nc"
	scene_file = "data/scene_id/scene_id_"+str(day)+"_res"+str(res)+"km_10comp.nc"

	#res = data_file.split("/")[-1].split(".")[0].split("_")[1]

	# Retrieve data
	preprocessed_data = load_data(data_file)
	rad_G16 = preprocessed_data['rad_G16_interp_corr'][:,:,[0,3,4,6,7,8]]
	rad_G18 = preprocessed_data['rad_G18_interp_corr'][:,:,[0,3,4,6,7,8]]
	#BT_G16 = npz_datafile['BT_G16_interp_corr'][:,:,[0,3,4,6,7,8]]
	#BT_G18 = npz_datafile['BT_G18_interp_corr'][:,:,[0,3,4,6,7,8]]
	for name, arr in [                                   
	    ("rad_G16", rad_G16),                            
	    ("rad_G18", rad_G18),                            
	    #("BT_G16", BT_G16),
	    #("BT_G18", BT_G18),
	]:
		print(
		    name,
		    arr.shape,
		    arr.dtype,
		    f"{arr.nbytes / 1024**3:.2f} GB"
		)
	# Flatten arrays then select channels
	rad_G16_flat = rad_G16.reshape(-1, rad_G16.shape[-1])
	rad_G18_flat = rad_G18.reshape(-1, rad_G18.shape[-1])
	#BT_G16_flat = BT_G16.reshape(-1, BT_G16.shape[-1])
	#BT_G18_flat = BT_G18.reshape(-1, BT_G18.shape[-1])

	#BT_mask = np.stack([
	#    BT_C08_interp_G16, BT_C11_interp_G16, BT_C12_interp_G16, BT_C14_interp_G16, BT_C15_interp_G16, BT_C16_interp_G16,
	#    BT_C08_interp_G18, BT_C11_interp_G18, BT_C12_interp_G18, BT_C14_interp_G18, BT_C15_interp_G18, BT_C16_interp_G18
	#], axis=-1)
	#mask = np.all((BT_mask > 0) & (BT_mask < 1e3) & np.isfinite(BT_mask), axis=1)

	#mask = np.ones(BT_G16_flat.shape[0], dtype=bool)
	#
	#for arr16, arr18 in zip(BT_G16_flat.T, BT_G18_flat.T):
	#	mask &= (arr16 > 0) & (arr16 < 1e3) & np.isfinite(arr16)
	#	mask &= (arr18 > 0) & (arr18 < 1e3) & np.isfinite(arr18)

	#mask = (
	#    (BT_mask[:, 0] > 0) & (BT_mask[:, 1] > 0) & (BT_mask[:, 2] > 0) & (BT_mask[:, 3] > 0) &
	#    (BT_mask[:, 4] > 0) & (BT_mask[:, 5] > 0) & (BT_mask[:, 6] > 0) & (BT_mask[:, 7] > 0) &
	#    (BT_mask[:, 8] > 0) & (BT_mask[:, 9] > 0) & (BT_mask[:, 10] > 0) & (BT_mask[:, 11] > 0) &
	#    (BT_mask[:, 0] < 1e03) & (BT_mask[:, 1] < 1e03) & (BT_mask[:, 2] < 1e03) & (BT_mask[:, 3] < 1e03) &
	#    (BT_mask[:, 4] < 1e03) & (BT_mask[:, 5] < 1e03) & (BT_mask[:, 6] < 1e03) & (BT_mask[:, 7] < 1e03) &
	#    (BT_mask[:, 8] < 1e03) & (BT_mask[:, 9] < 1e03) & (BT_mask[:, 10] < 1e03) & (BT_mask[:, 11] < 1e03) &
	#    ~np.isnan(BT_mask[:, 0]) & ~np.isnan(BT_mask[:, 1]) & ~np.isnan(BT_mask[:, 2]) & ~np.isnan(BT_mask[:, 3]) &
	#    ~np.isnan(BT_mask[:, 4]) & ~np.isnan(BT_mask[:, 5]) & ~np.isnan(BT_mask[:, 6]) & ~np.isnan(BT_mask[:, 7]) &
	#    ~np.isnan(BT_mask[:, 8]) & ~np.isnan(BT_mask[:, 9]) & ~np.isnan(BT_mask[:, 10]) & ~np.isnan(BT_mask[:, 11])
	#)
	#rad_mask = np.stack([
	#    rad_C08_interp_G16, rad_C11_interp_G16, rad_C12_interp_G16, rad_C14_interp_G16, rad_C15_interp_G16, rad_C16_interp_G16,
	#    rad_C08_interp_G18, rad_C11_interp_G18, rad_C12_interp_G18, rad_C14_interp_G18, rad_C15_interp_G18, rad_C16_interp_G18
	#], axis=-1)
	#mask_rad = np.all((rad_mask > 0) & (rad_mask < 1e3) & np.isfinite(rad_mask), axis=1)
	mask_rad = np.ones(rad_G16_flat.shape[0], dtype=bool)
	
	for arr16, arr18 in zip(rad_G16_flat.T, rad_G18_flat.T):
		mask_rad &= (arr16 > 0) & (arr16 < 1e3) & np.isfinite(arr16)
		mask_rad &= (arr18 > 0) & (arr18 < 1e3) & np.isfinite(arr18)
	#mask_rad = (
	#    (rad_mask[:, 0] > 0) & (rad_mask[:, 1] > 0) & (rad_mask[:, 2] > 0) & (rad_mask[:, 3] > 0) &
	#    (rad_mask[:, 4] > 0) & (rad_mask[:, 5] > 0) & (rad_mask[:, 6] > 0) & (rad_mask[:, 7] > 0) &
	#    (rad_mask[:, 8] > 0) & (rad_mask[:, 9] > 0) & (rad_mask[:, 10] > 0) & (rad_mask[:, 11] > 0) &
	#    (rad_mask[:, 0] < 1e03) & (rad_mask[:, 1] < 1e03) & (rad_mask[:, 2] < 1e03) & (rad_mask[:, 3] < 1e03) &
	#    (rad_mask[:, 4] < 1e03) & (rad_mask[:, 5] < 1e03) & (rad_mask[:, 6] < 1e03) & (rad_mask[:, 7] < 1e03) &
	#    (rad_mask[:, 8] < 1e03) & (rad_mask[:, 9] < 1e03) & (rad_mask[:, 10] < 1e03) & (rad_mask[:, 11] < 1e03) &
	#    ~np.isnan(rad_mask[:, 0]) & ~np.isnan(rad_mask[:, 1]) & ~np.isnan(rad_mask[:, 2]) & ~np.isnan(rad_mask[:, 3]) &
	#    ~np.isnan(rad_mask[:, 4]) & ~np.isnan(rad_mask[:, 5]) & ~np.isnan(rad_mask[:, 6]) & ~np.isnan(rad_mask[:, 7]) &
	#    ~np.isnan(rad_mask[:, 8]) & ~np.isnan(rad_mask[:, 9]) & ~np.isnan(rad_mask[:, 10]) & ~np.isnan(rad_mask[:, 11])
	#)
	for channel in range(6):
		for scene in range(10):
			print("Channel "+str(channel)+", scene "+str(scene))
			rad_G16 = rad_G16_flat[:,channel][mask_rad]#[mask & mask_rad] 
			rad_G18 = rad_G18_flat[:,channel][mask_rad]#[mask & mask_rad]
			lza_G16 = preprocessed_data['lza_G16_interp_corr'].ravel()[mask_rad]#[mask & mask_rad]
			lza_G18 = preprocessed_data['lza_G18_interp_corr'].ravel()[mask_rad]#[mask & mask_rad]

			# Retrieve scene labels
			scene_data = load_data(scene_file)
			labels = scene_data['labels'][mask_rad]#[mask & mask_rad]
			#npzfile_G16 = np.load(file_G16)
			#rad_G16 = npzfile_G16['arr_0'][:,channel]
			#TIR_G16 = npzfile_G16['arr_1'][:,channel]
			#labels_G16 = npzfile_G16['arr_2']
			#lza_G16 = npzfile_G16['arr_3']
			#npzfile_G18 = np.load(file_G18)
			#rad_G18 = npzfile_G18['arr_0'][:,channel]
			#TIR_G18 = npzfile_G18['arr_1'][:,channel]
			#labels_G18 = npzfile_G18['arr_2']
			#lza_G18 = npzfile_G18['arr_3']

			# Temperature difference
			#TIR_diff = TIR_G16 - TIR_G18
			# Radiance ratio
			#rad_ratio = rad_G16/rad_G18
			rad_ratio = np.divide(rad_G16, rad_G18, out=np.zeros_like(rad_G16), where=rad_G18!=0)
			print("Non-0 count:",np.count_nonzero(rad_G16),np.count_nonzero(rad_G18))
			print("NaN count:",np.count_nonzero(np.isnan(rad_G16)),np.count_nonzero(np.isnan(rad_G18)))
			# Find the unique labels
			unique_labels = np.unique(labels)
			scene_label = unique_labels[scene]
			scene_mask = (labels == scene_label)
			# Group values by label using a list comprehension
			rad_ratio_scene = rad_ratio[scene_mask]
			rad_G16_scene   = rad_G16[scene_mask]
			rad_G18_scene   = rad_G18[scene_mask]
			lza_G16_scene   = lza_G16[scene_mask]
			lza_G18_scene   = lza_G18[scene_mask]
			print("Shape (scene): ",rad_ratio_scene.shape)

			# Define x and y variables
			x_var = lza_G16_scene 
			y_var = rad_G16_scene

			## Define means per 5° bins
			#bins = np.arange(0,95,5)
			#bin_centers = np.mean(np.vstack([bins[:-1], bins[1:]]), axis=0)
			#indices = [i+1 for i in range(len(bin_centers))]
			##lza_G16_binned = np.digitize(lza_G16_slice,bins)
			##TIR_diff_lza_16_means = [np.mean(TIR_diff_slice_lza16[lza_G16_binned==index]) for index in indices]
			##lza_G18_binned = np.digitize(lza_G18_slice,bins)
			##TIR_diff_lza_18_means = [np.mean(TIR_diff_slice_lza18[lza_G18_binned==index]) for index in indices]
			#x_var_binned = np.digitize(x_var,bins)
			#means = np.array([np.mean(y_var[x_var_binned==index]) for index in indices])
			#sigma = np.array([np.std(y_var[x_var_binned==index]) for index in indices])
			#mask = (~np.isnan(means)) & (~np.isnan(sigma)) & (sigma>0)
			#means = means[mask] 
			#sigma = sigma[mask]
			#bin_centers = bin_centers[mask]

			# Fit ratio on all points
			popt, pcov = curve_fit(radiance_lin_ratio,(lza_G16_scene,lza_G18_scene),rad_ratio_scene)
			#popt, pcov = curve_fit(fitting_fct_1D,lza_G16_scene,rad_G16_scene)
			perr = np.sqrt(np.diag(pcov))	
			# Fit on binned data
			#popt, pcov = curve_fit(fitting_fct_1D,bin_centers,means,sigma=sigma)
			#perr = np.sqrt(np.diag(pcov))
			# Determine normalization factor for the ADM to integrate to unity
			norm_ADM = 1./(quad(radiance_integrand, 0, 90, args=(popt[0],))[0])
			print(norm_ADM)

			# Fit for the normalization in 1D
			#popt_norm, pcov_norm = curve_fit(radiance_lin_norm,(lza_G16_scene,np.full_like(lza_G16_scene,popt[0]),np.full_like(lza_G16_scene,popt[1])),rad_G16_scene)
			popt_norm, pcov_norm = curve_fit(radiance_lin_norm,(x_var,np.full_like(x_var,popt[0])),y_var)
			#popt_norm, pcov_norm = curve_fit(radiance_quad_norm,(x_var,np.full_like(x_var,popt[0]),np.full_like(x_var,popt[1])),y_var)
			perr_norm = np.sqrt(np.diag(pcov_norm))	

			print("a = "+str(popt_norm[0])+" +/- "+str(100*perr_norm[0]/np.abs(popt_norm[0]))+" %, b = "+str(popt[0])+" +/- "+str(100*perr[0]/np.abs(popt[0]))+" %")
			#flux_upward = (1./1000)*quad(flux_upward_integrand,0,90,args=(popt_norm,popt))[0]
			#flux = (1./1000)*quad(flux_integrand,0,90,args=(popt_norm,popt))[0]
			#print("Flux = "+str(flux)+" W/m2")

			#fig = plt.figure(figsize=(10, 8))
			#plt.plot(lza_G16_slice_lza18,rad_ratio_slice_lza18,'x',label="GOES 16/GOES 18 radiance, vza18 = 55°")
			#plt.plot(lza_G18_slice_lza16,rad_ratio_slice_lza16,'+',label="GOES 18/GOES 16 radiance, vza16 = 55°")
			#plt.plot(np.linspace(0,89,100),radiance_ratio((np.linspace(0,89,100),55),popt),label="Fit")
			#plt.xlabel(r"Viewing zenith angle ($\degree$)")
			#plt.ylabel(r"Radiance ratio")
			#plt.legend(frameon=False)
			#fig.tight_layout()
			#plt.savefig("figures/fit_ADM_slice_channel"+str(channel)+"_scene"+str(scene)+".png")
			#plt.show()

			if not args.no_plot:
				# Plot 1D data and fit
				fig = plt.figure(figsize=(10, 6))
				plt.scatter(x_var,y_var,s=0.1,c='b',label="GOES East data")
				plt.scatter(x_var,y_var/radiance_lin(x_var,*popt),s=0.1,c='r',label="Corrected GOES East data")
				plt.plot(np.linspace(0,89,100),radiance_lin_norm((np.linspace(0,89,100),*popt),popt_norm),label="Fit")
				#plt.plot(np.linspace(0,89,100),radiance_lin(np.linspace(0,89,100),*popt),label="Fit")
				#plt.tick_params('x', labelbottom=False)
				plt.ylabel(r"Radiance (mW/m$^{2}$.sr.$\mu$m)")
				plt.xlabel(r"Viewing zenith angle ($\degree$)")
				plt.xticks(np.arange(0, 100, 10))
				plt.legend(frameon=False)
				fig.tight_layout()
				plt.savefig("figures/ADM/fit_ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene"+str(scene)+".png")
				plt.show()

			# Save uncorrected and corrected radiances
			write_dataset(
				"data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene"+str(scene)+".nc",
				{"b": popt[0], "norm": norm_ADM},
				{"b": (), "norm": ()},
				attrs={"product": "ECO angular distribution model", "channel": channel, "scene": scene},
			)
