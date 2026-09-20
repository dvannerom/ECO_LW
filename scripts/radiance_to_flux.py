import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import math
import argparse
import psutil
import os
import gc
from netcdf_io import load_data, write_dataset
from adm import elmer, radiance_linear as radiance
from radiometry import radiance_to_brightness_temperature


ABI_CHANNELS = (0, 3, 4, 6, 7, 8)


def index_to_channel(index):
	return ABI_CHANNELS[index]

def Tbb(Tnb, c0, c1, c2, c3, c4, c5, c6):
	return c0 + c1*Tnb[0] + c2*Tnb[1] + c3*Tnb[2] + c4*Tnb[3] + c5*Tnb[4] + c6*Tnb[5]

if __name__ == '__main__':
    
	parser = argparse.ArgumentParser(description="sample argument parser")
	parser.add_argument("-d","--day", type=int, default=245)
	parser.add_argument("-r","--resolution", type=int, default=2)
	#parser.add_argument("-c","--channel", type=int, default=0)
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	args = parser.parse_args()

	day = args.day	
	res = args.resolution
	#channel = args.channel
	lambda_center = args.lambda_center

	radiance_file = "data/preprocessed_files/abi_"+str(day)+"_res"+str(res)+"km_step1.nc"
	scene_file = "data/scene_id/scene_id_"+str(day)+"_res"+str(res)+"km_10comp.nc"

	# Retrieve radiances
	radiance_data = load_data(radiance_file)
	print(
	    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
	    "GB"
	)
	lat_interp_grid = radiance_data['lat_interp_grid']
	lon_interp_grid = radiance_data['lon_interp_grid']
	shape_x, shape_y = radiance_data['lat_interp_grid'].shape[0], radiance_data['lat_interp_grid'].shape[1]
	print(
	    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
	    "GB"
	)
	lza_interp_grid_G16 = radiance_data['lza_G16_interp_corr'].ravel()
	lza_interp_grid_G18 = radiance_data['lza_G18_interp_corr'].ravel()
	print(
	    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
	    "GB"
	)

	# Retrieve scene labels
	scene_data = load_data(scene_file)
	labels = scene_data['labels']
	print(
	    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
	    "GB"
	)

	for channel in range(6):
		print("Channel "+str(channel))
		index_channel = index_to_channel(channel)
		planck_G16 = radiance_data['planck_G16'][index_channel,:]
		planck_G18 = radiance_data['planck_G18'][index_channel,:]
		print(
		    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
		    "GB"
		)
		rad_interp_G16 = radiance_data['rad_G16_interp_corr'][:,:,index_channel].ravel()
		rad_interp_G18 = radiance_data['rad_G18_interp_corr'][:,:,index_channel].ravel()
		print(
		    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
		    "GB"
		)

		# Retrieve ADMs
		b_scene = []
		norm_scene = []
		
		for scene in range(10):
			adm = load_data(
			    f"data/ADM/ADM_{day}_res{res}km_C{channel}_scene{scene}.nc"
			)
			
			b_scene.append(adm["b"])
			norm_scene.append(adm["norm"])
		print(
		    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
		    "GB"
		)

		#npzfile_scene0 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene0.npz")
		#npzfile_scene1 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene1.npz")
		#npzfile_scene2 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene2.npz")
		#npzfile_scene3 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene3.npz")
		#npzfile_scene4 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene4.npz")
		#npzfile_scene5 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene5.npz")
		#npzfile_scene6 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene6.npz")
		#npzfile_scene7 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene7.npz")
		#npzfile_scene8 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene8.npz")
		#npzfile_scene9 = np.load("data/ADM/ADM_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+"_scene9.npz")
		#b_scene = [npzfile_scene0['arr_0'],npzfile_scene1['arr_0'],npzfile_scene2['arr_0'],npzfile_scene3['arr_0'],npzfile_scene4['arr_0'],npzfile_scene5['arr_0'],npzfile_scene6['arr_0'],npzfile_scene7['arr_0'],npzfile_scene8['arr_0'],npzfile_scene9['arr_0']]
		#norm_scene = [npzfile_scene0['arr_1'],npzfile_scene1['arr_1'],npzfile_scene2['arr_1'],npzfile_scene3['arr_1'],npzfile_scene4['arr_1'],npzfile_scene5['arr_1'],npzfile_scene6['arr_1'],npzfile_scene7['arr_1'],npzfile_scene8['arr_1'],npzfile_scene9['arr_1']]
		
		#rad_G16_mask = np.stack([rad_G16_C14_interp,rad_G16_C11_interp,rad_G16_C15_interp], axis=-1)
		#mask_G16 = (rad_G16_mask[:, 0] != 0) & (rad_G16_mask[:, 1] != 0) & (rad_G16_mask[:, 2] != 0)
		#rad_G18_mask = np.stack([rad_G18_C14_interp,rad_G18_C11_interp,rad_G18_C15_interp], axis=-1)
		#mask_G18 = (rad_G18_mask[:, 0] != 0) & (rad_G18_mask[:, 1] != 0) & (rad_G18_mask[:, 2] != 0)

		# Convert G16 and G18 radiances to narrowband fluxes
		rad_interp_G16_corr = rad_interp_G16.copy()
		rad_interp_G18_corr = rad_interp_G18.copy()
		print(
		    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
		    "GB"
		)
		#for i in range(10):
		#	R_G16 = norm_scene[i]*radiance(lza_interp_grid_G16[labels==i],b_scene[i])
		#	rad_interp_G16_corr[labels==i] *= (1./R_G16)
		#	R_G18 = norm_scene[i]*radiance(lza_interp_grid_G18[labels==i],b_scene[i])
		#	rad_interp_G18_corr[labels==i] *= (1./R_G18)
		for i in range(10):
			mask = (labels == i)
			R_G16 = norm_scene[i] * radiance(lza_interp_grid_G16[mask],b_scene[i])
			rad_interp_G16_corr[mask] /= R_G16
			R_G18 = norm_scene[i] * radiance(lza_interp_grid_G18[mask],b_scene[i])
			rad_interp_G18_corr[mask] /= R_G18
		print(
		    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
		    "GB"
		)

		# Convert corrected radiances to BT
		BT_G16 = radiance_to_brightness_temperature(rad_interp_G16_corr, planck_G16)
		BT_G18 = radiance_to_brightness_temperature(rad_interp_G18_corr, planck_G18)
		print(
		    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
		    "GB"
		)

		#plt.hist(rad_interp_G16_corr[(rad_interp_G16_corr>0) & (rad_interp_G16_corr<1e03)],100)
		#plt.savefig("figures/rad_G16_"+str(res)+"_C"+str(channel)+".png")
		#plt.close()
		#plt.hist(rad_interp_G18_corr[(rad_interp_G18_corr>0) & (rad_interp_G18_corr<1e03)],100)
		#plt.savefig("figures/rad_G18_"+str(res)+"_C"+str(channel)+".png")
		#plt.close()

		#rad_interp_G16[(rad_interp_G16<=0) | (rad_interp_G16>1e03) | (lon_interp_grid>-75) | (lon_interp_grid<-135)] = np.nan
		#rad_interp_G18[(rad_interp_G18<=0) | (rad_interp_G18>1e03) | (lon_interp_grid>-75) | (lon_interp_grid<-135)] = np.nan
		#rad_interp_G16_corr[(rad_interp_G16_corr<=0) | (rad_interp_G16_corr>1e03) | (lon_interp_grid>-75) | (lon_interp_grid<-135)] = np.nan
		#rad_interp_G18_corr[(rad_interp_G18_corr<=0) | (rad_interp_G18_corr>1e03) | (lon_interp_grid>-75) | (lon_interp_grid<-135)] = np.nan
		diff = rad_interp_G16 - rad_interp_G18
		diff_corr = rad_interp_G16_corr - rad_interp_G18_corr
		#mask1 = (np.abs(diff)<200)
		#mask2 = (np.abs(diff)<200) & (lza_interp_grid_G16<75) & (lza_interp_grid_G18<75)
		#mask1_corr = (np.abs(diff_corr)<200)
		#mask2_corr = (np.abs(diff_corr)<200) & (lza_interp_grid_G16<75) & (lza_interp_grid_G18<75)
		print(str(np.nanstd(diff/rad_interp_G16))+"\t"+str(np.nanstd(diff_corr/rad_interp_G16_corr)))

		#bins = np.linspace(-200,200,100)
		#plt.hist(diff_corr,bins)
		#plt.yscale('log')
		#plt.savefig("figures/radiance_to_flux_diff_CH"+str(channel)+".png")
		#plt.close()

		diff = diff.reshape(shape_x,shape_y) 
		diff_corr = diff_corr.reshape(shape_x,shape_y) 

		#max_diff = max(np.nanmax(diff),-np.nanmin(diff))
		#max_diff_corr = max(np.nanmax(diff_corr),-np.nanmin(diff_corr))
		#max_glob = max(max_diff,max_diff_corr)
		print(
		    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
		    "GB"
		)

		## Delete unnecessary arrays (for memory efficiency)
		#del rad_interp_G16
		#del rad_interp_G18
		#
		#del rad_interp_G16_corr
		#del rad_interp_G18_corr
		#
		#del lza_interp_grid_G16
		#del lza_interp_grid_G18
		#
		#del labels
		#
		#gc.collect()

		## Draw labels on original map
		#lon_min = -153
		#lon_max = -59
		#fig, axs = plt.subplots(1, 2, figsize=(14, 8), subplot_kw={'projection': ccrs.Sinusoidal(central_longitude=lambda_center)})
		#pc0 = axs[0].pcolormesh(lon_interp_grid, lat_interp_grid, diff, cmap='bwr', transform=ccrs.PlateCarree(), vmin=-40, vmax=40)
		#axs[0].set_global()
		#axs[0].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
		#axs[0].coastlines()
		#axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
		#cbar = fig.colorbar(pc0, ax=axs[0], orientation="vertical")
		#print(
		#    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
		#    "GB"
		#)
		#pc1 = axs[1].pcolormesh(lon_interp_grid, lat_interp_grid, diff_corr, cmap='bwr', transform=ccrs.PlateCarree(), vmin=-40, vmax=40)
		#axs[1].set_global()
		#axs[1].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
		#axs[1].coastlines()
		#axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
		#cbar = fig.colorbar(pc1, ax=axs[1], orientation="vertical")
		##cbar.set_label("Scene ID")
		#fig.tight_layout()
		#plt.savefig("figures/radiance_to_flux_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+".png")
		#plt.close()
		#print(
		#    psutil.Process(os.getpid()).memory_info().rss / 1024**3,
		#    "GB"
		#)

		write_dataset(
			"data/narrowband_flux/narrowband_flux_"+str(day)+"_res"+str(res)+"km_C"+str(channel)+".nc",
			{
				"BT_G16": BT_G16.reshape(shape_x, shape_y),
				"BT_G18": BT_G18.reshape(shape_x, shape_y),
				"lat": lat_interp_grid,
				"lon": lon_interp_grid,
			},
			{
				"BT_G16": ("y", "x"),
				"BT_G18": ("y", "x"),
				"lat": ("y", "x"),
				"lon": ("y", "x"),
			},
			attrs={"product": "ECO narrowband brightness temperature flux input", "channel": channel},
		)

		del planck_G16
		del planck_G18

		del rad_interp_G16
		del rad_interp_G18
		
		del rad_interp_G16_corr
		del rad_interp_G18_corr

		del BT_G16
		del BT_G18
		
		gc.collect()
