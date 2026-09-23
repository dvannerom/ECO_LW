import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
from sklearn.preprocessing import PolynomialFeatures
import psutil
import os
from broadband import cubic_regression
from netcdf_io import load_data, write_dataset

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

	# Only lat/lon are needed here; the preprocessed file also holds several
	# (y, x, channel) radiance/BT/lza variables (~3.8 GB each) that must not be
	# loaded into memory for this step.
	preprocessed_data = load_data(
		"data/preprocessed_files/abi_"+str(day)+"_res"+str(res)+"km_step1.nc",
		variable_names=("lat_interp_grid", "lon_interp_grid"),
	)
	shape_x, shape_y = preprocessed_data['lat_interp_grid'].shape[0], preprocessed_data['lat_interp_grid'].shape[1]
	lat_interp_grid = preprocessed_data['lat_interp_grid']
	lon_interp_grid = preprocessed_data['lon_interp_grid']
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

	n_pixels = shape_x * shape_y

	def load_satellite_bt(satellite):
		"""Load one satellite's 6-channel BT into a single preallocated (n_pixels, 6) array.

		Only the requested BT_{satellite} variable is read from each per-channel
		file (not the sibling satellite's BT or lat/lon), and satellites are
		loaded one at a time by the caller so only one (n_pixels, 6) array is
		ever resident.
		"""
		rad = np.empty((n_pixels, 6), dtype=np.float32)
		var_name = f"BT_{satellite}"
		for ch in range(6):
			f = load_data(
				f"data/narrowband_flux/narrowband_flux_{day}_res{res}km_C{ch}.nc",
				variable_names=(var_name,),
			)
			rad[:, ch] = f[var_name].ravel()
		np.nan_to_num(rad, copy=False)
		return rad

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
	# coeff_cubic/intercept_cubic now live in src/broadband.py and are applied via cubic_regression()


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

	# Convert BT to flux, one satellite at a time so only one (n_pixels, 6)
	# radiance array and its Tbb are ever resident (peak halves vs. holding
	# both satellites' radiances simultaneously).
	sigma = 5.670374E-08

	rad_G16 = load_satellite_bt("G16")
	Tbb_G16 = cubic_regression(rad_G16)
	del rad_G16
	flux_G16 = sigma*np.power(Tbb_G16,4)
	del Tbb_G16
	print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")

	rad_G18 = load_satellite_bt("G18")
	Tbb_G18 = cubic_regression(rad_G18)
	del rad_G18
	flux_G18 = sigma*np.power(Tbb_G18,4)
	del Tbb_G18
	print(psutil.Process(os.getpid()).memory_info().rss / 1024**3,"GB")

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

	write_dataset(
		"data/broadband_flux/broadband_flux_"+str(day)+"_res"+str(res)+"km.nc",
		{
			"flux_G16": flux_G16,
			"flux_G18": flux_G18,
			"lat": lat_interp_grid,
			"lon": lon_interp_grid,
		},
		{
			"flux_G16": ("y", "x"),
			"flux_G18": ("y", "x"),
			"lat": ("y", "x"),
			"lon": ("y", "x"),
		},
		attrs={"product": "ECO daily broadband flux"},
	)
