"""Shared GaussianMixture subclass used by the scene-ID pipeline.

Lives in ``src/`` (rather than in ``scripts/train_GMM.py``) so that pickled
pipelines reference an importable module instead of ``__main__``.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits


class ParallelGaussianMixture(GaussianMixture):
	"""GaussianMixture with concurrent random initializations sharing X."""

	def __init__(
		self, n_components=1, *, covariance_type="full", tol=1e-3,
		reg_covar=1e-6, max_iter=100, n_init=1, init_params="kmeans",
		weights_init=None, means_init=None, precisions_init=None,
		random_state=None, warm_start=False, verbose=0, verbose_interval=10,
		n_jobs=1,
	):
		super().__init__(
			n_components=n_components, covariance_type=covariance_type, tol=tol,
			reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
			init_params=init_params, weights_init=weights_init,
			means_init=means_init, precisions_init=precisions_init,
			random_state=random_state, warm_start=warm_start,
			verbose=verbose, verbose_interval=verbose_interval,
		)
		self.n_jobs = n_jobs

	def fit(self, X, y=None):
		if self.n_init <= 1 or self.n_jobs <= 1:
			return super().fit(X, y)

		random_state = check_random_state(self.random_state)
		seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)

		def fit_one(seed):
			model = GaussianMixture(
				n_components=self.n_components,
				covariance_type=self.covariance_type,
				tol=self.tol,
				reg_covar=self.reg_covar,
				max_iter=self.max_iter,
				n_init=1,
				init_params=self.init_params,
				weights_init=self.weights_init,
				means_init=self.means_init,
				precisions_init=self.precisions_init,
				random_state=int(seed),
				warm_start=False,
				verbose=self.verbose,
				verbose_interval=self.verbose_interval,
			)
			with threadpool_limits(limits=1):
				return model.fit(X, y)

		with ThreadPoolExecutor(max_workers=min(self.n_jobs, self.n_init)) as executor:
			models = list(executor.map(fit_one, seeds))
		best = max(models, key=lambda model: model.lower_bound_)
		for attribute in (
			"weights_",
			"means_",
			"covariances_",
			"precisions_",
			"precisions_cholesky_",
			"converged_",
			"n_iter_",
			"lower_bound_",
			"n_features_in_",
			"feature_names_in_",
		):
			if hasattr(best, attribute):
				setattr(self, attribute, getattr(best, attribute))
		return self


def load_pipeline(model_file):
	"""Load a joblib-pickled scene-ID pipeline.

	Models trained before this class moved out of ``scripts/train_GMM.py`` were
	pickled with module ``__main__``; expose the class there so they still load.
	"""
	main_module = sys.modules["__main__"]
	if not hasattr(main_module, "ParallelGaussianMixture"):
		main_module.ParallelGaussianMixture = ParallelGaussianMixture
	return joblib.load(model_file)
