import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
import time
import math
import argparse
import joblib
from scene_features import build_scene_features

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
	"""
	Create a heatmap from a numpy array and two lists of labels.
	
	Parameters
	----------
	data
	    A 2D numpy array of shape (M, N).
	row_labels
	    A list or array of length M with the labels for the rows.
	col_labelsd
	    A list or array of length N with the labels for the columns.
	ax
	    A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
	    not provided, use current Axes or create a new one.  Optional.
	cbar_kw
	    A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
	cbarlabel
	    The label for the colorbar.  Optional.
	**kwargs
	    All other arguments are forwarded to `imshow`.
	"""
	
	if ax is None:
		ax = plt.gca()
	
	if cbar_kw is None:
		cbar_kw = {}
	
	# Plot the heatmap
	im = ax.imshow(data, **kwargs)#, vmin=-10, vmax=10)
	
	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
	
	# Show all ticks and label them with the respective list entries.
	ax.set_xticks(range(data.shape[1]), labels=col_labels, rotation=30, ha="right", rotation_mode="anchor")
	ax.set_yticks(range(data.shape[0]), labels=row_labels)
	
	# Let the horizontal axes labeling appear on top.
	ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
	
	# Turn spines off and create white grid.
	ax.spines[:].set_visible(False)
	
	ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
	ax.tick_params(which="minor", bottom=False, left=False)
	
	return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
	"""
	A function to annotate a heatmap.
	
	Parameters
	----------
	im
	    The AxesImage to be labeled.
	data
	    Data used to annotate.  If None, the image's data is used.  Optional.
	valfmt
	    The format of the annotations inside the heatmap.  This should either
	    use the string format method, e.g. "$ {x:.2f}", or be a
	    `matplotlib.ticker.Formatter`.  Optional.
	textcolors
	    A pair of colors.  The first is used for values below a threshold,
	    the second for those above.  Optional.
	threshold
	    Value in data units according to which the colors from textcolors are
	    applied.  If None (the default) uses the middle of the colormap as
	    separation.  Optional.
	**kwargs
	    All other arguments are forwarded to each call to `text` used to create
	    the text labels.
	"""
	
	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()
	
	# Normalize the threshold to the images color range.
	if threshold is not None:
		threshold = im.norm(threshold)
	else:
		threshold = im.norm(data.max())/2.
	
	# Set default alignment to center, but allow it to be
	# overwritten by textkw.
	kw = dict(horizontalalignment="center",verticalalignment="center")
	kw.update(textkw)
	
	# Get the formatter in case a string is supplied
	if isinstance(valfmt, str):
		valfmt = mpl.ticker.StrMethodFormatter(valfmt)
	
	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
			#condition = int(abs(data[i, j]) < threshold)
			#kw.update(color=textcolors[condition])
			text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
			texts.append(text)
	
	return texts

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="GOES16/18 Scene ID with optional PCA+GMM")
	parser.add_argument("-f", "--input_file", type=str, nargs="+", default="data/preprocessed_files/abi_pix1000_step5.nc")
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	parser.add_argument("-n", "--n_components", type=int, default=10, help="Number of GMM components")
	parser.add_argument("--n_components_file", type=str, default=None,
		help="JSON file from find_nComponents.py; overrides -n with its selected n_components")
	parser.add_argument("--use_pca", action="store_true", help="Enable PCA before GMM")
	parser.add_argument("--pca_var", type=float, default=0.98, help="Cumulative variance to keep (e.g., 0.98)")
	args = parser.parse_args()
	
	input_file = args.input_file
	lambda_center = args.lambda_center
	n_components = args.n_components
	if args.n_components_file:
		selection = json.load(open(args.n_components_file))
		if selection.get("n_components") is None:
			raise SystemExit(f"{args.n_components_file} has no selected n_components (find_nComponents.py found no passing candidate)")
		n_components = int(selection["n_components"])
		print(f"Using n_components={n_components} from {args.n_components_file}")
	use_pca = args.use_pca
	pca_var = args.pca_var
	
	os.makedirs("figures/scene_id", exist_ok=True)

	BT_arrays = []

	for f in input_file:	
		BT_arrays.append(build_scene_features(f, pixel_step=3)[0])

	BT_merged = np.concatenate(BT_arrays, axis=0)

	print(BT_merged.shape)
	
	# ---------------------------------------
	# Build pipeline Scale -> PCA (optional) -> GMM
	# ---------------------------------------
	
	steps = [
		("scaler", StandardScaler())
	]
	
	if use_pca:
		# 1) Fit a full PCA for diagnostics
		pca_full = PCA(svd_solver='full')
		pca_full.fit(StandardScaler().fit_transform(BT_merged))
		
		# Auto-select number of PCs to reach target variance
		cvar = np.cumsum(pca_full.explained_variance_ratio_)
		n_pc = int(np.searchsorted(cvar, pca_var) + 1)
		print(f"Running PCA with {n_pc} PCs")

		steps.append(
			("pca",
				PCA(
					n_components=n_pc,
					svd_solver="full",
					whiten=False)
			)
		)

	steps.append(
		("gmm",
			GaussianMixture(
				n_components=n_components,
				n_init=5,
				covariance_type="full",
				random_state=42)
		)
	)

	pipeline = Pipeline(steps)
	pipeline.fit(BT_merged)
	labels_nozeros = pipeline.predict(BT_merged)
	
	if use_pca:
		# Scree plot
		fig, ax = plt.subplots(figsize=(7, 4))
		ax.plot(np.arange(1, len(cvar)+1), cvar, marker='o')
		ax.axhline(pca_var, color='r', linestyle='--', label=f"Target var={pca_var:.2f}")
		ax.axvline(n_pc, color='g', linestyle=':', label=f"Selected PCs={n_pc}")
		ax.set_xlabel("Principal component")
		ax.set_ylabel("Cumulative explained variance")
		ax.legend(frameon=False)
		fig.tight_layout()
		plt.savefig(f"figures/scene_id/pca_scree_merged_2km.png", dpi=150)
		plt.close(fig)
		
		# Loadings (first min(6, n_pc) PCs)
		# loadings shape: [original_features x PCs]
		pca = pipeline.named_steps["pca"]
		loadings = pca.components_.T  # columns are PCs
		pc_to_show = min(6, loadings.shape[1])
		#feature_names = [
		#    "C08_G16","C11_G16","C12_G16","C14_G16","C15_G16","C16_G16",
		#    "C08_G18","C11_G18","C12_G18","C14_G18","C15_G18","C16_G18",
		#    #"BTD_C08","BTD_C11","BTD_C12","BTD_C14","BTD_C15","BTD_C16",
		#    "BTD14-11_G16","BTD14-11_G18","BTD14-15_G16","BTD14-15_G18"
		#]
		feature_names = [
		    "C08_av","C11_av","C12_av","C14_av","C15_av","C16_av",
		    "BTD14-11","BTD14-15","BTD08-14","BTD16-14"
		]
		fig, axs = plt.subplots(pc_to_show, 1, figsize=(8, 2.2*pc_to_show), sharex=True)
		axs = np.atleast_1d(axs)
		for i in range(pc_to_show):
			axs[i].bar(np.arange(len(feature_names)), loadings[:, i])
			axs[i].set_title(f"PC{i+1} loadings")
			axs[i].grid(True, alpha=0.3)
		axs[-1].set_xticks(np.arange(len(feature_names)))
		axs[-1].set_xticklabels(feature_names, rotation=45, ha='right')
		fig.tight_layout()
		plt.savefig(f"figures/scene_id/pca_loadings_merged_2km.png", dpi=150)
		plt.close(fig)

	components_bins = ["0","1","2","3","4","5","6","7","8","9"]
	#features_bins = ["C08_G16","C11_G16","C12_G16","C14_G16","C15_G16","C16_G16",
	#		         "C08_G18","C11_G18","C12_G18","C14_G18","C15_G18","C16_G18"]
	#features_diff_bins = ["BTD14-11_G16","BTD14-11_G18","BTD14-15_G16","BTD14-15_G18"]
	#features_diff_calib_bins = ["BTD_C08","BTD_C11","BTD_C12","BTD_C14","BTD_C15","BTD_C16"]

	features_bins = ["C08_av","C11_av","C12_av","C14_av","C15_av","C16_av"]
	features_diff_bins = ["BTD14-11","BTD14-15","BTD08-14","BTD16-14"]

	centroids = np.empty((10,6,))
	centroids_diff = np.empty((10,4,))
	#centroids_diff_calib = np.empty((10,6,))

	# -----------------------------
	# Physically meaningful label ordering
	# -----------------------------
	# Sort by original-space mean of C14_G16 (index 2 in BT stacking)
	# gmm.means_ are in Z-space; map to original feature space:
	#  (1) if PCA used: inverse PCA, then inverse scale
	#  (2) else: inverse scale directly
	scaler = pipeline.named_steps["scaler"]
	gmm = pipeline.named_steps["gmm"]
	
	if use_pca:
		pca = pipeline.named_steps["pca"]
		means_orig = scaler.inverse_transform(pca.inverse_transform(gmm.means_))
	else:
		means_orig = scaler.inverse_transform(gmm.means_)
	
	sort_idx = np.argsort(means_orig[:, 3])  # 2 == C14_G16
	label_map = {orig: new for new, orig in enumerate(sort_idx)}
	
	for i in range(10):
		for j in range(6):
			centroids[label_map[i],j] = np.mean(BT_merged[:,j][labels_nozeros==i])
		#for j in range(6):
		#	centroids_diff_calib[label_map[i],j] = np.mean(BT_nozeros[:,j+12][labels_nozeros==i])
		for j in range(4):
			centroids_diff[label_map[i],j] = np.mean(BT_merged[:,j+6][labels_nozeros==i])

	# Plot centroids
	fig, ax = plt.subplots(figsize=(10, 6))
	im, cbar = heatmap(centroids, components_bins, features_bins, ax=ax, cmap="rainbow", cbarlabel="BT (K)")
	texts = annotate_heatmap(im, valfmt="{x:.1f}")
	ax.set_xlabel("Feature")
	ax.set_ylabel("Scene ID")
	fig.tight_layout()
	suffix = f"merged_res2km_{int(n_components)}comp"
	plt.savefig(f"figures/centroids_{suffix}.png", dpi=150)
	# Plot diff
	fig, ax = plt.subplots(figsize=(4, 6))
	im, cbar = heatmap(centroids_diff, components_bins, features_diff_bins, ax=ax, cmap=mpl.cm.bwr, cbarlabel="BTD")
	texts = annotate_heatmap(im, valfmt="{x:.1f}",threshold=-6)
	ax.set_xlabel("Feature")
	ax.set_ylabel("Scene ID")
	fig.tight_layout()
	plt.savefig(f"figures/centroids_diff_{suffix}.png", dpi=150)
	## Plot diff calib
	#fig, ax = plt.subplots(figsize=(6, 6))
	#im, cbar = heatmap(centroids_diff_calib, components_bins, features_diff_calib_bins, ax=ax, cmap=mpl.cm.bwr, cbarlabel="BTD")
	#texts = annotate_heatmap(im, valfmt="{x:.1f}",threshold=-6)
	#ax.set_xlabel("Feature")
	#ax.set_ylabel("Scene ID")
	#fig.tight_layout()
	#suffix = f"{day}_{res}_{int(n_components)}comp"
	#plt.savefig(f"figures/centroids_diff_calib_{suffix}.png", dpi=150)

	# -----------------------------
	# Save model
	# -----------------------------
	model_path = f"data/models/gmm_pipeline_{suffix}.joblib"
	os.makedirs("data/models", exist_ok=True)
	joblib.dump(pipeline, model_path)
	print(f"Model saved to {model_path}")
	
#	# -----------------------------
#	# Save outputs (labels + settings)
#	# -----------------------------
#	out_npz_path = f"data/scene_id/scene_id_{suffix}.npz"
#	np.savez(out_npz_path,
#	         labels=labels,
#	         labels_nozeros=labels_nozeros,
#	         sorted_labels=sorted_labels,
#	         use_pca=use_pca,
#	         n_pc=(int(n_pc) if use_pca else None),
#	         pca_var=(float(pca_var) if use_pca else None),
#	         n_components=int(n_components),
#	         random_state=42)
