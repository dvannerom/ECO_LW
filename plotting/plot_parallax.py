import numpy as np
from netcdf_io import load_data
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import math
import argparse
from geospatial import normalize_longitude

to_minus180_180 = normalize_longitude

if __name__ == '__main__':
    
	parser = argparse.ArgumentParser(description="sample argument parser")
	parser.add_argument("-f","--input_file", type=str)
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	args = parser.parse_args()
	
	input_file = args.input_file
	lambda_center = args.lambda_center

	# Retrieve data
	dataset = load_data(input_file)
	lat_interp_grid = dataset['lat_interp_grid']
	lon_interp_grid = dataset['lon_interp_grid']
	lon_new = to_minus180_180(lon_interp_grid.copy())
	rad_C14_interp_G16 = dataset['rad_G16_interp'][:,:,2]
	rad_C14_interp_G16_corr = dataset['rad_G16_interp_corr'][:,:,2]
	rad_C14_interp_G18 = dataset['rad_G18_interp'][:,:,2]
	rad_C14_interp_G18_corr = dataset['rad_G18_interp_corr'][:,:,2]

	# Mask NaNs
	#rad_C14_interp_G16 = rad_C14_interp_G16[~np.isnan(rad_C14_interp_G16)]
	#rad_C14_interp_G16_corr = rad_C14_interp_G16_corr[~np.isnan(rad_C14_interp_G16_corr)]
	#rad_C14_interp_G18 = rad_C14_interp_G18[~np.isnan(rad_C14_interp_G18)]
	#rad_C14_interp_G18_corr = rad_C14_interp_G18_corr[~np.isnan(rad_C14_interp_G18_corr)]

	fig, axs = plt.subplots(1, 2, figsize=(14, 8))
	im0 = axs[0].imshow(lat_interp_grid)
	fig.colorbar(im0, ax=axs[0], orientation='vertical', pad=0.05, shrink=0.8)
	im1 = axs[1].imshow(lon_interp_grid)#, vmin=-175, vmax=-50)
	fig.colorbar(im1, ax=axs[1], orientation='vertical', pad=0.05, shrink=0.8)
	plt.savefig("figures/lat_lon_grid.png")
	plt.close()

	#mask = (rad_C14_interp_G16 == 0) | (rad_C14_interp_G18 == 0)
	#rad_C14_interp_G16[mask] = np.nan
	#rad_C14_interp_G18[mask] = np.nan
	#mask_corr = (rad_C14_interp_G16_corr == 0) | (rad_C14_interp_G18_corr == 0)
	#rad_C14_interp_G16_corr[mask_corr] = np.nan
	#rad_C14_interp_G18_corr[mask_corr] = np.nan

	# Longitude restriction
	#lon_mask = (lon_interp_grid < -80) & (lon_interp_grid > -130)
	#lat_interp_grid = lat_interp_grid[lon_mask]
	#lon_interp_grid = lon_interp_grid[lon_mask]
	#rad_C14_interp_G16 = rad_C14_interp_G16[lon_mask]
	#rad_C14_interp_G16_corr = rad_C14_interp_G16_corr[lon_mask]
	#rad_C14_interp_G18 = rad_C14_interp_G18[lon_mask]
	#rad_C14_interp_G18_corr = rad_C14_interp_G18_corr[lon_mask]

	# Define differences
	diff = rad_C14_interp_G16 - rad_C14_interp_G18
	diff_corr = rad_C14_interp_G16_corr - rad_C14_interp_G18_corr
	diff_diff = diff-diff_corr
	print(np.nanstd(diff.flatten()),np.nanstd(diff_corr.flatten()))

	max_diff = max(np.nanmax(diff),-np.nanmin(diff))
	max_diff_corr = max(np.nanmax(diff_corr),-np.nanmin(diff_corr))
	max_glob = max(max_diff,max_diff_corr)

	# Draw diff histograms before and after corrections
	fig = plt.figure(figsize=(10, 6))
	#bins = np.linspace(-100,100,200)
	plt.hist(diff.flatten(),100,alpha=0.5,color='b')
	plt.hist(diff_corr.flatten(),100,alpha=0.5,color='r')
	plt.yscale('log')
	fig.tight_layout()
	plt.savefig("figures/parallax_hist.png")
	plt.close()

	# Draw parallax effect before and after correction
	cmap = mpl.cm.bwr
#	fig, axs = plt.subplots(1, 2, figsize=(14, 8), subplot_kw={'projection': ccrs.Sinusoidal(central_longitude=lambda_center)})
#	pc0 = axs[0].pcolormesh(lon_interp_grid, lat_interp_grid, rad_C14_interp_G18, transform=ccrs.PlateCarree(), vmin=0, vmax=35)
#	axs[0].set_global()
#	axs[0].coastlines()
#	gl0 = axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
#	cbar = fig.colorbar(pc0, ax=axs[0], orientation="vertical", shrink=0.7)
#	pc1 = axs[1].pcolormesh(lon_interp_grid, lat_interp_grid, rad_C14_interp_G16, transform=ccrs.PlateCarree(), vmin=0, vmax=35)
#	axs[1].set_global()
#	axs[1].coastlines()
#	gl1 = axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
#	cbar = fig.colorbar(pc1, ax=axs[1], orientation="vertical", shrink=0.7)
	lon_min = -151
	lon_max = -61
	fig, axs = plt.subplots(1, 3, figsize=(14, 8), subplot_kw={'projection': ccrs.Sinusoidal(central_longitude=lambda_center)})
	pc0 = axs[0].pcolormesh(lon_interp_grid, lat_interp_grid, diff, cmap=cmap, transform=ccrs.PlateCarree(), vmin=-15, vmax=15)
	axs[0].set_global()
	axs[0].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
	axs[0].coastlines()
	gl0 = axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
	cbar = fig.colorbar(pc0, ax=axs[0], orientation="vertical", shrink=0.7)
	pc1 = axs[1].pcolormesh(lon_interp_grid, lat_interp_grid, diff_corr, cmap=cmap, transform=ccrs.PlateCarree(), vmin=-15, vmax=15)
	axs[1].set_global()
	axs[1].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
	axs[1].coastlines()
	gl1 = axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
	cbar = fig.colorbar(pc1, ax=axs[1], orientation="vertical", shrink=0.7)
	pc2 = axs[2].pcolormesh(lon_interp_grid, lat_interp_grid, diff_diff, cmap=cmap, transform=ccrs.PlateCarree(), vmin=-5, vmax=5)
	axs[2].set_global()
	axs[2].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
	axs[2].coastlines()
	gl2 = axs[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
	cbar = fig.colorbar(pc2, ax=axs[2], orientation="vertical", shrink=0.7)
	fig.tight_layout()
	plt.savefig("figures/parallax.png")
	plt.show()
