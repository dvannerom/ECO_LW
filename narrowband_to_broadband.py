import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
from sklearn.preprocessing import PolynomialFeatures
import psutil
import os

def rad_to_T(rad,planck_fk1,planck_fk2,planck_bc1,planck_bc2):
	ratio = planck_fk1/rad
	a = ratio + 1
	b = np.log(a)
	c = planck_fk2/b
	d = c - planck_bc1
	return d/planck_bc2

def cubic_regression(rad, coeff, intercept):

	x0 = rad[:, 0]
	x1 = rad[:, 1]
	x2 = rad[:, 2]
	x3 = rad[:, 3]
	x4 = rad[:, 4]
	x5 = rad[:, 5]
	
	x = [x0, x1, x2, x3, x4, x5]
	
	out = np.full(rad.shape[0], intercept, dtype=rad.dtype)
	
	k = 0
	
	# Linear
	for i in range(6):
		out += coeff[k] * x[i]
		k += 1
	
	# Quadratic
	for i in range(6):
		xi = x[i]
		
		for j in range(i, 6):
			out += coeff[k] * xi * x[j]
			k += 1
	
	# Cubic
	for i in range(6):
		xi = x[i]
		
		for j in range(i, 6):
			xij = xi * x[j]
			
			for l in range(j, 6):
				out += coeff[k] * xij * x[l]
				k += 1
	
	return out

if __name__ == '__main__':
    
	parser = argparse.ArgumentParser(description="sample argument parser")
	parser.add_argument("-d","--day", type=int, default=245)
	parser.add_argument("-r","--resolution", type=int, default=2)
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	args = parser.parse_args()
	
	day = args.day	
	res = args.resolution
	lambda_center = args.lambda_center

	print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")

	npzfile = np.load("data/preprocessed_files/abi_"+str(day)+"_res"+str(res)+"km_step1.npz")
	shape_x, shape_y = npzfile['lat_interp_grid'].shape[0], npzfile['lat_interp_grid'].shape[1]
	lat_interp_grid = npzfile['lat_interp_grid']
	lon_interp_grid = npzfile['lon_interp_grid']
	#lza_interp_grid_G16 = npzfile['lza_G16_interp_corr']
	#lza_interp_grid_G18 = npzfile['lza_G18_interp_corr']
	print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")

	# Retrieve radiances
	#npzfile_CH0 = np.load("data/narrowband_flux/narrowband_flux_"+str(day)+"_res"+str(res)+"km_C0.npz")
	#rad_G16_CH0 = npzfile_CH0['arr_0']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#rad_G18_CH0 = npzfile_CH0['arr_1']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#npzfile_CH1 = np.load("data/narrowband_flux/narrowband_flux_"+str(day)+"_res"+str(res)+"km_C1.npz")
	#rad_G16_CH1 = npzfile_CH1['arr_0']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#rad_G18_CH1 = npzfile_CH1['arr_1']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#npzfile_CH2 = np.load("data/narrowband_flux/narrowband_flux_"+str(day)+"_res"+str(res)+"km_C2.npz")
	#rad_G16_CH2 = npzfile_CH2['arr_0']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#rad_G18_CH2 = npzfile_CH2['arr_1']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#npzfile_CH3 = np.load("data/narrowband_flux/narrowband_flux_"+str(day)+"_res"+str(res)+"km_C3.npz")
	#rad_G16_CH3 = npzfile_CH3['arr_0']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#rad_G18_CH3 = npzfile_CH3['arr_1']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#npzfile_CH4 = np.load("data/narrowband_flux/narrowband_flux_"+str(day)+"_res"+str(res)+"km_C4.npz")
	#rad_G16_CH4 = npzfile_CH4['arr_0']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#rad_G18_CH4 = npzfile_CH4['arr_1']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#npzfile_CH5 = np.load("data/narrowband_flux/narrowband_flux_"+str(day)+"_res"+str(res)+"km_C5.npz")
	#rad_G16_CH5 = npzfile_CH5['arr_0']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)
	#rad_G18_CH5 = npzfile_CH5['arr_1']# + np.random.normal(0, 1.2, npzfile_CH0['arr_0'].shape)

	#rad_G16 = np.stack([rad_G16_CH0,rad_G16_CH1,rad_G16_CH2,rad_G16_CH3,rad_G16_CH4,rad_G16_CH5],axis=1)
	#rad_G16 = np.nan_to_num(rad_G16)
	#rad_G18 = np.stack([rad_G18_CH0,rad_G18_CH1,rad_G18_CH2,rad_G18_CH3,rad_G18_CH4,rad_G18_CH5],axis=1)
	#rad_G18 = np.nan_to_num(rad_G18)

	rad_G16 = []
	rad_G18 = []
	
	for ch in range(6):
		with np.load(
		    f"data/narrowband_flux/narrowband_flux_{day}_res{res}km_C{ch}.npz"
		) as f:
		    rad_G16.append(f["arr_0"].astype(np.float32))
		    rad_G18.append(f["arr_1"].astype(np.float32))
	
	print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")
	rad_G16 = np.stack(rad_G16, axis=1)
	rad_G18 = np.stack(rad_G18, axis=1)

	np.nan_to_num(rad_G16, copy=False)
	np.nan_to_num(rad_G18, copy=False)

	# GOES channels 08, 11, 12, 14, 15, 16
	#a = np.array([4.63213883e-03,5.05040878e-03,5.27956064e-03,0.00645188,0.00693793,0.00632944])
	#b = np.array([7.99453640e+00,6.07740875e+00,5.41373985e+00,4.75053341,4.38290917,4.10502937])
	#T_G16 = np.power(rad_G16,1./b)/a
	#T_G18 = np.power(rad_G18,1./b)/a

	# rad_to_T coefficients
	#planck_fk1 = np.asarray([5.06871e+04,1.97799e+04,1.34321e+04,8.51022e+03,6.45462e+03,5.10127e+03])
	#planck_fk2 = np.asarray([2.33158e+03,1.70383e+03,1.49761e+03,1.28627e+03,1.17303e+03,1.08453e+03])
	#planck_bc1 = np.asarray([1.55228,0.18733,0.09102,0.22516,0.21702,0.06266])
	#planck_bc2 = np.asarray([0.99667,0.99948,0.99971,0.99920,0.99916,0.99974])
	#T_G16 = rad_to_T(rad_G16,planck_fk1,planck_fk2,planck_bc1,planck_bc2)
	#T_G18 = rad_to_T(rad_G18,planck_fk1,planck_fk2,planck_bc1,planck_bc2)

	#bins = np.linspace(-1,1,100)
	#plt.hist(T_G16[:,0],100)
	#plt.hist(T_G18[:,0],100)
	#plt.savefig("figures/BT_CH08.png")
	#plt.close()
	#plt.hist(T_G16[:,1],100)
	#plt.hist(T_G18[:,1],100)
	#plt.savefig("figures/BT_CH11.png")
	#plt.close()
	#plt.hist(T_G16[:,2],100)
	#plt.hist(T_G18[:,2],100)
	#plt.savefig("figures/BT_CH12.png")
	#plt.close()
	#plt.hist(T_G16[:,3],100)
	#plt.hist(T_G18[:,3],100)
	#plt.savefig("figures/BT_CH14.png")
	#plt.close()
	#plt.hist(T_G16[:,4],100)
	#plt.hist(T_G18[:,4],100)
	#plt.savefig("figures/BT_CH15.png")
	#plt.close()
	#plt.hist(T_G16[:,5],100)
	#plt.hist(T_G18[:,5],100)
	#plt.savefig("figures/BT_CH16.png")
	#plt.close()

	#poly2 = PolynomialFeatures(degree=2, include_bias=False)
	#rad_G16_poly2 = poly2.fit_transform(rad_G16)
	#rad_G18_poly2 = poly2.fit_transform(rad_G18)

	#poly3 = PolynomialFeatures(degree=3, include_bias=False)
	#rad_G16_poly3 = poly3.fit_transform(rad_G16)
	#rad_G18_poly3 = poly3.fit_transform(rad_G18)
	#print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")

	coeff_linear = [0.20605798,0.01805503,0.14840809,0.25119058,0.07625716,0.15832402]
	intercept_linear = 35.34936733219382
	coeff_quadratic = [-7.29180244e-02,-1.43889949e+00,-5.95981211e-02,-7.30620941e-01,
	                   1.87368418e+00,1.08928752e+00,-2.65951108e-03,7.31794082e-03,
	                   -3.59304621e-04,-1.74492348e-02,8.15521662e-03,8.61979179e-03,
	                   -5.20139236e-03,3.25113765e-03,-1.63710263e-02,3.15207810e-02,
	                   -8.21125990e-03,1.49315785e-03,-1.91976988e-02,1.71826356e-02,
	                   -3.12825477e-03,1.04778670e-01,-2.27570602e-01,7.11314751e-02,
	                   1.17296322e-01,-6.87044855e-02,-1.42712364e-03]
	intercept_quadratic = 56.29448115395326
	coeff_cubic = [4.69962675e+00,1.62183406e+01,7.61262687e-01,-8.92166533e+00,
	               -4.86682683e+00,-1.14231072e+01,-9.45069025e-02,2.50272570e-01,
	               -6.50158635e-02,-2.75717730e-01,-2.60760940e-02,2.61120108e-01,
	               -1.52914394e-01,1.53894060e-01,-4.17800377e-01,1.34917427e+00,
	               -1.18213776e+00,-3.57863341e-02,2.01244030e-03,-2.27775285e-01,
	               2.00477535e-01,1.35122229e+00,-3.38625494e+00,1.45998620e+00,
	               1.33664170e+00,-3.39776272e-01,-1.42809106e-01,1.34300203e-04,
	               -5.90140657e-04,5.23752905e-05,6.85399660e-04,-7.85379265e-05,
	               -8.49371329e-05,-1.12610983e-03,-8.95732635e-05,-9.87184197e-04,
	               3.26488292e-03,1.53438709e-04,1.54233887e-04,-8.07753393e-04,
	               8.23816754e-04,-5.41931708e-05,5.54989968e-03,-1.15234035e-02,
	               2.01203872e-03,4.83026715e-03,-1.93513714e-03,-5.30714915e-04,
	               -5.90870290e-03,-1.42607367e-03,4.87879796e-02,-3.76451896e-02,
	               9.61241946e-03,3.47400352e-04,3.55688764e-03,-8.14456940e-05,
	               -1.92873252e-03,-1.08799800e-01,1.45713496e-01,-2.63262126e-02,
	               -4.05851615e-02,1.82810953e-03,6.20333119e-03,9.08135558e-05,
	               -1.11355453e-03,9.35698960e-04,-4.23572660e-04,1.61386994e-03,
	               -8.46903041e-03,4.58617016e-03,4.59084130e-03,-2.15894325e-03,
	               -2.58627281e-04,6.59718996e-02,-1.07547359e-01,6.20092029e-03,
	               3.84030469e-02,2.54580565e-02,-1.21409230e-02,1.93717934e-03,
	               -1.79004995e-02,6.84997236e-03,1.47793639e-04]
	intercept_cubic = 384.61181077479637


#	coeff_linear = np.asarray([16.02115098,-0.80979908,3.08614319,3.56242094,1.92214979,9.03824266])
#	intercept_linear = 53.70355670053132
#
	#coeff_quadratic = np.asarray([8.69617009,21.49524107,3.9535179,-29.55871389,22.11273294,
	#                              24.09425929,-4.3941757,-0.46544739,-0.05427172,-0.89696627,
	#                              1.20687803,2.60877684,6.66351333,-0.80399884,-11.60460897,
	#                              4.97936156,-1.19772383,0.8532276,-1.1873563,1.51498901,
	#                              -1.25248116,7.19264461,-9.73852146,7.42501236,4.32266726,
	#                              -7.22289746,-0.35978324])
	#intercept_quadratic = 24.404397131227142

	#coeff_cubic = np.asarray([-4.32944086e+01,-5.27089900e+01,-3.99654726e+00,4.42709576e+01,
    #                          -2.92026211e+01,7.50369384e+01,-3.00160305e+01,-1.34535431e+01,
    #                          2.72610023e-02,7.26973796e+00,-6.20660041e+00,4.28036709e+01,
    #                          -4.89382685e+01,7.69584661e+00,7.14554308e+01,-2.93634987e+01,
    #                          2.13704244e+01,-5.72220201e-01,-1.04198083e+01,5.59951006e+00,
    #                          5.22813770e+00,-1.07622260e+01,-1.14017286e+01,-6.57870864e+00,
    #                          1.29975872e+01,3.19698426e+00,-2.15014816e+01,2.66123845e+00,
    #                          7.29296333e-01,-7.13843398e-01,7.81088903e-01,-1.44063711e+00,
    #                          2.42856485e+00,1.61228445e+00,-4.91637230e-01,-4.89071611e+00,
    #                          2.36767971e+00,3.67735797e+00,1.33613528e+00,-2.27346288e+00,
    #                          3.12441569e+00,-3.17994208e+00,4.60526224e+00,-6.06958169e+00,
    #                          -1.05192143e+00,2.15555263e+00,3.84856662e-01,-3.31468652e+00,
    #                          -9.74845329e+00,1.92788555e+00,3.16718604e+01,-2.19114523e+01,
    #                          1.64048058e+01,-4.24990425e-01,-4.41548845e+00,3.39209864e+00,
    #                          -2.61678018e+00,-3.20093182e+01,3.82499762e+01,-1.91208420e+01,
    #                          -7.95789048e+00,-8.84847351e-01,4.70891224e+00,4.84359592e-01,
    #                          -7.77972668e-01,1.06599661e+00,-1.38434121e+00,3.60398654e+00,
    #                          -5.90001814e+00,5.04441311e+00,2.26418915e+00,-3.50422175e+00,
    #                          5.92149231e-01,9.25544978e+00,-1.18654519e+01,-1.92921883e+00,
    #                          -1.70115322e-01,1.92773280e+01,-9.21021072e+00,2.43785475e+00,
    #                          -1.14752472e+01,6.27685131e+00,1.37858688e+00])
	#intercept_cubic = (-0.3233792631885706)

	# Convert BT to flux
	sigma = 5.670374E-08
	#Tbb_G16 = rad_G16 @ coeff_linear + intercept_linear
	#Tbb_G16 = rad_G16_poly2 @ coeff_quadratic + intercept_quadratic
	#Tbb_G16 = rad_G16_poly3 @ coeff_cubic + intercept_cubic
	Tbb_G16 = cubic_regression(rad_G16,coeff_cubic,intercept_cubic)
	print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")
	flux_G16 = sigma*np.power(Tbb_G16,4)
	#Tbb_G18 = rad_G18 @ coeff_linear + intercept_linear
	#Tbb_G18 = rad_G18_poly2 @ coeff_quadratic + intercept_quadratic
	#Tbb_G18 = rad_G18_poly3 @ coeff_cubic + intercept_cubic
	Tbb_G18 = cubic_regression(rad_G18,coeff_cubic,intercept_cubic)
	flux_G18 = sigma*np.power(Tbb_G18,4)
	print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")
	
	del rad_G16, rad_G18
	del Tbb_G16, Tbb_G18

	#plt.hist(flux_G16[(flux_G16>50) & (flux_G16<1e03)],100)
	#plt.savefig("figures/flux_G16.png")
	#plt.close()
	#plt.hist(flux_G18[(flux_G18>50) & (flux_G18<1e03)],100)
	#plt.savefig("figures/flux_G18.png")
	#plt.close()

	#flux_G16 = rad_G16 @ coeff_linear + intercept_linear
	#flux_G18 = rad_G18 @ coeff_linear + intercept_linear
	#flux_G16 = rad_G16_poly2 @ coeff_quadratic + intercept_quadratic
	#flux_G18 = rad_G18_poly2 @ coeff_quadratic + intercept_quadratic
	#flux_G16 = rad_G16_poly3 @ coeff_cubic + intercept_cubic
	#flux_G18 = rad_G18_poly3 @ coeff_cubic + intercept_cubic

	flux_G16[(flux_G16<50) | (flux_G16>1e03)] = np.nan
	print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")
	flux_G18[(flux_G18<50) | (flux_G18>1e03)] = np.nan
	print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")
	#flux_G16[(flux_G16<50) | (flux_G16>1e03) | (lza_interp_grid_G16.ravel()>70) | (lza_interp_grid_G18.ravel()>70)] = np.nan
	#flux_G18[(flux_G18<50) | (flux_G18>1e03) | (lza_interp_grid_G16.ravel()>70) | (lza_interp_grid_G18.ravel()>70)] = np.nan
	flux_G16 = flux_G16.reshape(shape_x,shape_y)
	flux_G18 = flux_G18.reshape(shape_x,shape_y)
	diff = flux_G16-flux_G18
	print(str(np.nanstd(diff.ravel()))+"\t"+str(np.nanstd(diff.ravel()/flux_G16.ravel()))+"\t"+str(np.nanstd(diff.ravel()/flux_G18.ravel())))

	#plt.hist(diff.ravel(),100)
	#plt.savefig("figures/flux_diff.png")
	#plt.close()

	max_flux = 380#max(np.nanmax(flux_G16),np.nanmax(flux_G18))
	min_flux = 80#min(np.nanmin(flux_G16[flux_G16>0]),np.nanmin(flux_G18[flux_G18>0]))
	print(min_flux,max_flux)
	max_diff = 40#max(np.nanmax(diff),-np.nanmin(diff))
	print(max_diff)

	## Draw labels on original map
	#lon_min = -163
	#lon_max = -49
	#fig, axs = plt.subplots(1, 3, figsize=(22, 8), subplot_kw={'projection': ccrs.Sinusoidal(central_longitude=lambda_center)})
	#pc0 = axs[0].pcolormesh(lon_interp_grid, lat_interp_grid, flux_G16, cmap='jet', transform=ccrs.PlateCarree(), vmin=min_flux, vmax=max_flux)
	#axs[0].set_global()
	#axs[0].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
	#axs[0].coastlines()
	#axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
	#cbar0 = fig.colorbar(pc0, ax=axs[0], orientation="vertical")
	#pc1 = axs[1].pcolormesh(lon_interp_grid, lat_interp_grid, flux_G18, cmap='jet', transform=ccrs.PlateCarree(), vmin=min_flux, vmax=max_flux)
	#axs[1].set_global()
	#axs[1].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
	#axs[1].coastlines()
	#axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
	#cbar1 = fig.colorbar(pc1, ax=axs[1], orientation="vertical")
	#pc2 = axs[2].pcolormesh(lon_interp_grid, lat_interp_grid, diff, cmap='bwr', transform=ccrs.PlateCarree(), vmin=-max_diff, vmax=max_diff)
	#axs[2].set_global()
	#axs[2].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
	#axs[2].coastlines()
	#axs[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
	#cbar2 = fig.colorbar(pc2, ax=axs[2], orientation="vertical")
	##cbar.set_label("Scene ID")
	#fig.tight_layout()
	#plt.savefig("figures/broadband_flux_"+str(day)+"_res"+str(res)+"km.png")
	#plt.close()

	np.savez("data/broadband_flux/broadband_flux_"+str(day)+"_res"+str(res)+"km",flux_G16,flux_G18)
