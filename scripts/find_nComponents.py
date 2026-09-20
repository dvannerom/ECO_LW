import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
from joblib import Parallel, delayed
from scipy.spatial.distance import jensenshannon
import math
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import argparse
from netcdf_io import load_data

def evaluate_gmm(X, n_components):
	gmm = GaussianMixture(n_components=n_components, n_init=5, random_state=42)
	gmm.fit(X)
	labels = gmm.predict(X)

	bic = gmm.bic(X)
	aic = gmm.aic(X)

	## Convert to GPU (cuDF DataFrame)
	#X_gpu = cudf.DataFrame(X)
	#labels_gpu = cudf.Series(labels)
	#
	## Compute silhouette on GPU
	#sil_score = silhouette_score(X_gpu, labels_gpu)
	#sil = silhouette_score(X, labels)

	## Compute posterior probabilities (responsibilities)
	#responsibilities = gmm.predict_proba(X)
	#
	## Compute entropy term (avoid log(0) by adding small epsilon)
	#entropy = np.sum(responsibilities * np.log(responsibilities + 1e-10))
	#
	## ICL = BIC - 2 * entropy
	#icl = bic - 2 * entropy

	return n_components, bic, aic#, icl

if __name__ == '__main__':
    
	parser = argparse.ArgumentParser(description="sample argument parser")
	parser.add_argument("-f","--input_file", type=str, default="data/preprocessed_files/abi_pix1000_step5.nc")
	parser.add_argument("--use_pca", action="store_true", help="Enable PCA before GMM")
	parser.add_argument("--pca_var", type=float, default=0.95, help="Cumulative variance to keep (e.g., 0.98)")
	args = parser.parse_args()
	
	input_file = args.input_file
	use_pca = args.use_pca
	pca_var = args.pca_var

	day = input_file.split("/")[-1].split(".")[0].split("_")[1]
	res = input_file.split("/")[-1].split(".")[0].split("_")[2]

	# Retrieve data
	dataset = load_data(input_file)
	lza_interp_grid_G16 = np.cos(np.radians(dataset['lza_G16_interp'])).flatten()
	lza_interp_grid_G18 = np.cos(np.radians(dataset['lza_G18_interp'])).flatten()
	BT_C08_interp_G16 = dataset['BT_G16_interp'][:,:,0].flatten()
	BT_C11_interp_G16 = dataset['BT_G16_interp'][:,:,3].flatten()
	BT_C12_interp_G16 = dataset['BT_G16_interp'][:,:,4].flatten()
	BT_C14_interp_G16 = dataset['BT_G16_interp'][:,:,6].flatten()
	BT_C15_interp_G16 = dataset['BT_G16_interp'][:,:,7].flatten()
	BT_C16_interp_G16 = dataset['BT_G16_interp'][:,:,8].flatten()
	BT_C08_interp_G18 = dataset['BT_G18_interp'][:,:,0].flatten()
	BT_C11_interp_G18 = dataset['BT_G18_interp'][:,:,3].flatten()
	BT_C12_interp_G18 = dataset['BT_G18_interp'][:,:,4].flatten()
	BT_C14_interp_G18 = dataset['BT_G18_interp'][:,:,6].flatten()
	BT_C15_interp_G18 = dataset['BT_G18_interp'][:,:,7].flatten()
	BT_C16_interp_G18 = dataset['BT_G18_interp'][:,:,8].flatten()

	# Define averages
	BT_C08_av = (BT_C08_interp_G16 + BT_C08_interp_G18)/2
	BT_C11_av = (BT_C11_interp_G16 + BT_C11_interp_G18)/2
	BT_C12_av = (BT_C12_interp_G16 + BT_C12_interp_G18)/2
	BT_C14_av = (BT_C14_interp_G16 + BT_C14_interp_G18)/2
	BT_C15_av = (BT_C15_interp_G16 + BT_C15_interp_G18)/2
	BT_C16_av = (BT_C16_interp_G16 + BT_C16_interp_G18)/2
	BT_diff1 = BT_C14_av - BT_C11_av
	BT_diff2 = BT_C14_av - BT_C15_av

	# Define difference arrays
	BT_diff_C08 = BT_C08_interp_G16 - BT_C08_interp_G18
	BT_diff_C11 = BT_C11_interp_G16 - BT_C11_interp_G18
	BT_diff_C12 = BT_C12_interp_G16 - BT_C12_interp_G18
	BT_diff_C14 = BT_C14_interp_G16 - BT_C14_interp_G18
	BT_diff_C15 = BT_C15_interp_G16 - BT_C15_interp_G18
	BT_diff_C16 = BT_C16_interp_G16 - BT_C16_interp_G18
	BT_diff1_G16 = BT_C14_interp_G16 - BT_C11_interp_G16
	BT_diff2_G16 = BT_C14_interp_G16 - BT_C15_interp_G16
	BT_diff1_G18 = BT_C14_interp_G18 - BT_C11_interp_G18
	BT_diff2_G18 = BT_C14_interp_G18 - BT_C15_interp_G18

	# Reshape to get final BT array
	#BT = np.stack([
	#    BT_C08_interp_G16, BT_C11_interp_G16, BT_C12_interp_G16, BT_C14_interp_G16, BT_C15_interp_G16, BT_C16_interp_G16,
	#    BT_C08_interp_G18, BT_C11_interp_G18, BT_C12_interp_G18, BT_C14_interp_G18, BT_C15_interp_G18, BT_C16_interp_G18,
	#    #BT_diff_C08, BT_diff_C11, BT_diff_C12, BT_diff_C14, BT_diff_C15, BT_diff_C16,
	#    BT_diff1_G16, BT_diff1_G18, BT_diff2_G16, BT_diff2_G18
	#], axis=-1).reshape(-1, 16)
	BT = np.stack([
	    BT_C08_av, BT_C11_av, BT_C12_av, BT_C14_av, BT_C15_av, BT_C16_av,
	    BT_diff1, BT_diff2
	], axis=-1).reshape(-1, 8)
	
	# Mask invalids
	BT_mask = np.stack([
	    BT_C08_interp_G16, BT_C11_interp_G16, BT_C12_interp_G16, BT_C14_interp_G16, BT_C15_interp_G16, BT_C16_interp_G16,
	    BT_C08_interp_G18, BT_C11_interp_G18, BT_C12_interp_G18, BT_C14_interp_G18, BT_C15_interp_G18, BT_C16_interp_G18
	], axis=-1)
	
	mask = (
	    (BT_mask[:, 0] > 0) & (BT_mask[:, 1] > 0) & (BT_mask[:, 2] > 0) & (BT_mask[:, 3] > 0) &
	    (BT_mask[:, 4] > 0) & (BT_mask[:, 5] > 0) & (BT_mask[:, 6] > 0) & (BT_mask[:, 7] > 0) &
	    (BT_mask[:, 8] > 0) & (BT_mask[:, 9] > 0) & (BT_mask[:, 10] > 0) & (BT_mask[:, 11] > 0) &
	    (BT_mask[:, 0] < 1e03) & (BT_mask[:, 1] < 1e03) & (BT_mask[:, 2] < 1e03) & (BT_mask[:, 3] < 1e03) &
	    (BT_mask[:, 4] < 1e03) & (BT_mask[:, 5] < 1e03) & (BT_mask[:, 6] < 1e03) & (BT_mask[:, 7] < 1e03) &
	    (BT_mask[:, 8] < 1e03) & (BT_mask[:, 9] < 1e03) & (BT_mask[:, 10] < 1e03) & (BT_mask[:, 11] < 1e03) &
	    ~np.isnan(BT_mask[:, 0]) & ~np.isnan(BT_mask[:, 1]) & ~np.isnan(BT_mask[:, 2]) & ~np.isnan(BT_mask[:, 3]) &
	    ~np.isnan(BT_mask[:, 4]) & ~np.isnan(BT_mask[:, 5]) & ~np.isnan(BT_mask[:, 6]) & ~np.isnan(BT_mask[:, 7]) &
	    ~np.isnan(BT_mask[:, 8]) & ~np.isnan(BT_mask[:, 9]) & ~np.isnan(BT_mask[:, 10]) & ~np.isnan(BT_mask[:, 11])
	)

	BT_nozeros = BT[mask]

	# Normalize the data
	scaler = StandardScaler()
	BT_nozeros_scaled = scaler.fit_transform(BT_nozeros)

	# Define range of components to test
	n_components_range = range(2,20)

	# Prepare variables we will use later for label sorting and saving
	Z = BT_nozeros_scaled
	pca = None
	n_pc = None
	
	if use_pca:
		# 1) Fit a full PCA for diagnostics
		pca_full = PCA(svd_solver='full')
		pca_full.fit(BT_nozeros_scaled)
		
		# Auto-select number of PCs to reach target variance
		cvar = np.cumsum(pca_full.explained_variance_ratio_)
		n_pc = int(np.searchsorted(cvar, pca_var) + 1)
		print("Running PCA with "+str(n_pc)+" pc")	
	
		# 2) Fit reduced PCA with n_pc and transform data
		pca = PCA(n_components=n_pc, svd_solver='full', whiten=False)
		Z = pca.fit_transform(BT_nozeros_scaled)

	# Parallel execution
	with tqdm_joblib(tqdm(total=len(n_components_range))) as progress_bar:
		#results = np.asarray(Parallel(verbose=10,n_jobs=len(n_components_range),batch_size=1)(delayed(evaluate_gmm)(BT_nozeros_scaled, k) for k in n_components_range))
		results = np.asarray(Parallel(verbose=10,n_jobs=len(n_components_range),batch_size=1)(delayed(evaluate_gmm)(Z, k) for k in n_components_range))
	bic_scores = results[:,1]
	aic_scores = results[:,2]

	# Gradient of BIC scores
	bic_grad = np.gradient(bic_scores)

	# Normalize lists
	aic_scores = (aic_scores - min(aic_scores))/(max(aic_scores) - min(aic_scores))
	bic_scores = (bic_scores - min(bic_scores))/(max(bic_scores) - min(bic_scores))
	bic_grad = -(bic_grad - min(bic_grad))/(max(bic_grad) - min(bic_grad))
	#silhouette_scores = (silhouette_scores - min(silhouette_scores))/(max(silhouette_scores) - min(silhouette_scores))

	# Precompute euclidean distances
	#distances = pairwise_distances(BT_nozeros_scaled)

	# Split data set into two (randpmly) to apply JS distance evaluation
	#indices = np.random.permutation(BT_nozeros_scaled.shape[0])
	#half = int(BT_nozeros_scaled.shape[0]/2)
	#indices1 = indices[:half]
	#indices2 = indices[half:]
	#data1 = BT_nozeros_scaled[indices1]
	#data2 = BT_nozeros_scaled[indices2]
	
	## Fit GMM for each number of components and compute BIC and AIC
	#for n in tqdm(n_components_range):
	#	gmm = GaussianMixture(n_components=n, n_init=5, random_state=42)
	#	gmm.fit(BT_nozeros_scaled)
	#	aic_scores.append(gmm.aic(BT_nozeros_scaled))
	#	bic_scores.append(gmm.bic(BT_nozeros_scaled))
	#	labels= gmm.fit_predict(BT_nozeros_scaled)
	#	#silhouette_scores.append(silhouette_score(BT_nozeros_scaled, labels))
	#	#silhouette_scores.append(silhouette_score(distances, labels, metric="precomputed"))
	#	## ICL approximation: BIC - sum of entropy of responsibilities
	#	#log_resp = gmm.predict_proba(BT_nozeros_scaled)
	#	#entropy = -np.sum(log_resp * np.log(log_resp + 1e-10))  # add small value to avoid log(0)
	#	#icl_scores.append(gmm.bic(BT_nozeros_scaled) + entropy)
	#	## Jensen-Shannon Distance
	#	#gmm_JS1 = GaussianMixture(n_components=n, n_init=10, random_state=42)
	#	#gmm_JS2 = GaussianMixture(n_components=n, n_init=10, random_state=42)
	#	#gmm_JS1.fit(data1)
	#	#gmm_JS2.fit(data2)
	#	#p = np.exp(gmm_JS1.score_samples(data1))
	#	#q = np.exp(gmm_JS2.score_samples(data2))
	#	#js = jensenshannon(p, q)
	#	#js_distances.append(js)

	#icl_scores = (icl_scores - min(icl_scores))/(max(icl_scores) - min(icl_scores))
	#js_distances = (js_distances - min(js_distances))/(max(js_distances) - min(js_distances))

	# Plot BIC and AIC scores
	plt.figure(figsize=(12, 6))
	plt.plot(n_components_range, aic_scores, label='AIC', marker='x')
	plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
	plt.plot(n_components_range, bic_grad, label='BIC grad', marker='v')
	#plt.plot(n_components_range, silhouette_scores, label='Silhouette', marker='*')
	#plt.plot(n_components_range, icl_scores, label='ICL', marker='P')
	#plt.plot(n_components_range, js_distances, label='JS', marker='s')
	plt.xlabel('Number of Components')
	plt.ylabel('Score')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("figures/gmm_scores_4BT_2BTD_2Sat_"+day+"_"+res+".png")
	plt.show()
