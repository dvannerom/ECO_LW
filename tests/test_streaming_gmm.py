"""Full-data and bounded-memory regression tests for streaming mixture EM."""

import unittest
from unittest.mock import patch
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import streaming_gmm as streaming


def features(rows=800, seed=53):
    rng = np.random.RandomState(seed)
    latent = rng.normal(size=(rows, 4))
    latent[:, 0] += rng.choice([-3.0, 3.0], size=rows)
    mixing = rng.normal(size=(4, 10))
    return latent @ mixing + 0.1 * rng.normal(size=(rows, 10))


def factory(data, size=93):
    return lambda: (data[start:start + size] for start in range(0, len(data), size))


class StreamingGMMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        cls.data = features()

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def test_preprocessing_matches_dense_and_constant_features(self):
        for data in (
            self.data, np.column_stack((self.data[:, :9], np.ones(len(self.data)))),
            self.data[:2], self.data[:7],
        ):
            with self.subTest(rows=len(data), constant=np.ptp(data[:, -1]) == 0):
                count, mean, scatter = streaming._stream_moments(factory(data))
                scaler, pca = streaming._preprocessing(count, mean, scatter)
                dense_scaler = StandardScaler().fit(data)
                dense_pca = PCA(n_components=0.98, svd_solver="full").fit(
                    dense_scaler.transform(data)
                )
                np.testing.assert_allclose(scaler.mean_, dense_scaler.mean_, atol=1e-13)
                np.testing.assert_allclose(scaler.var_, dense_scaler.var_, rtol=1e-13)
                np.testing.assert_allclose(scaler.scale_, dense_scaler.scale_, rtol=1e-13)
                self.assertEqual(pca.n_components_, dense_pca.n_components_)
                self.assertEqual(scaler.n_samples_seen_, len(data))
                for attribute in (
                    "components_", "explained_variance_", "explained_variance_ratio_",
                    "singular_values_", "noise_variance_",
                ):
                    np.testing.assert_allclose(
                        getattr(pca, attribute), getattr(dense_pca, attribute),
                        rtol=1e-11, atol=1e-12,
                    )
                np.testing.assert_allclose(
                    pca.transform(scaler.transform(data)),
                    dense_pca.transform(dense_scaler.transform(data)), atol=1e-12,
                )

    def test_centered_moments_large_offsets(self):
        data = 1e10 + self.data
        count, mean, scatter = streaming._stream_moments(factory(data, 17))
        centered = data - data[0]
        expected_mean = data[0] + centered.mean(axis=0)
        centered -= centered.mean(axis=0)
        self.assertEqual(count, len(data))
        np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=2e-5)
        np.testing.assert_allclose(scatter, centered.T @ centered, rtol=2e-6, atol=2e-3)

    def test_weighted_statistics_and_dense_em_objective(self):
        count, mean, scatter = streaming._stream_moments(factory(self.data))
        scaler, pca = streaming._preprocessing(count, mean, scatter)
        transformed = pca.transform(scaler.transform(self.data))
        dimensions = transformed.shape[1]
        model = GaussianMixture(n_components=2, covariance_type="full")
        streaming._set_parameters(
            model, np.array([0.4, 0.6]), transformed[[8, 250]],
            np.repeat(np.eye(dimensions)[None, :, :], 2, axis=0),
        )
        masses, means, scatters, objective = streaming._expectation_statistics(
            factory(self.data, 31), scaler, pca, model, count,
        )
        log_prob, log_resp = model._estimate_log_prob_resp(transformed)
        responsibilities = np.exp(log_resp)
        dense_mass = responsibilities.sum(axis=0)
        dense_mean = responsibilities.T @ transformed / dense_mass[:, None]
        dense_scatter = np.array([
            (transformed - dense_mean[k]).T
            @ ((transformed - dense_mean[k]) * responsibilities[:, k, None])
            for k in range(2)
        ])
        np.testing.assert_allclose(masses, dense_mass, rtol=1e-13)
        np.testing.assert_allclose(means, dense_mean, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(scatters, dense_scatter, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(objective, log_prob.mean(), places=12)
        dense = GaussianMixture(n_components=2, covariance_type="full")
        dense._m_step(transformed, log_resp)
        streaming._maximization(model, masses, means, scatters, count)
        for attribute in ("weights_", "means_", "covariances_", "precisions_cholesky_"):
            np.testing.assert_allclose(
                getattr(model, attribute), getattr(dense, attribute), rtol=1e-11, atol=1e-12,
            )
        self.assertAlmostEqual(
            model.score_samples(transformed).mean(),
            dense.score_samples(transformed).mean(), places=11,
        )

    def test_full_dense_em_with_identical_initialization(self):
        data_factory = factory(self.data, 59)
        model = streaming.fit_streaming_gmm(
            data_factory, 2, 123, initializations=1, tol=1e-7,
        )
        scaler, pca = model[0], model[1]
        transformed = pca.transform(scaler.transform(self.data))
        start_seed = np.random.RandomState(123).randint(np.iinfo(np.int32).max)
        centers = streaming._initial_centers(
            data_factory, scaler, pca, len(self.data), 2, np.random.RandomState(start_seed),
        )
        covariance = np.diag(pca.explained_variance_ * ((len(self.data) - 1) / len(self.data)))
        covariance.flat[::len(covariance) + 1] += 1e-6
        dense = GaussianMixture(n_components=2)
        streaming._set_parameters(
            dense, np.array([0.5, 0.5]), centers,
            np.repeat(covariance[None, :, :], 2, axis=0),
        )
        for _ in range(model[-1].n_iter_):
            _, log_resp = dense._estimate_log_prob_resp(transformed)
            dense._m_step(transformed, log_resp)
        np.testing.assert_allclose(
            model.score_samples(self.data), dense.score_samples(transformed),
            rtol=1e-11, atol=1e-11,
        )
        np.testing.assert_allclose(model[-1].covariances_, dense.covariances_, atol=1e-11)

    def test_partition_invariance_reproducibility_and_best_start(self):
        first = streaming.fit_streaming_gmm(factory(self.data, 27), 2, 42, tol=1e-6)
        second = streaming.fit_streaming_gmm(factory(self.data, 311), 2, 42, tol=1e-6)
        repeated = streaming.fit_streaming_gmm(factory(self.data, 27), 2, 42, tol=1e-6)
        for other in (second, repeated):
            np.testing.assert_allclose(
                first.score_samples(self.data), other.score_samples(self.data), atol=1e-10,
            )
            np.testing.assert_allclose(first[-1].means_, other[-1].means_, atol=1e-10)
        self.assertTrue(first[-1].converged_)
        self.assertEqual(first[-1].n_init, 2)
        self.assertEqual(first[-1].streaming_initialization_scores_.shape, (2,))
        self.assertAlmostEqual(
            first[-1].lower_bound_, max(first[-1].streaming_initialization_scores_), places=12,
        )
        self.assertAlmostEqual(
            first[-1].lower_bound_, first.score_samples(self.data).mean(), places=12,
        )
        self.assertTrue(np.all(np.diff(first[-1].lower_bounds_) >= -1e-9))

    def test_independent_initializations_match_joint_fit_and_report_progress(self):
        combined = streaming.fit_streaming_gmm(
            factory(self.data, 41), 2, 42, initializations=2, tol=1e-6,
        )
        split_models = []
        for initialization in range(2):
            progress = []
            model = streaming.fit_streaming_gmm(
                factory(self.data, 41), 2, 42, initializations=2, tol=1e-6,
                initialization_index=initialization,
                progress_callback=lambda start, iteration, objective, converged:
                    progress.append((start, iteration, objective, converged)),
            )
            split_models.append(model)
            self.assertTrue(progress)
            self.assertEqual(progress[-1][0], initialization)
            self.assertTrue(progress[-1][3])
            self.assertTrue(all(event[1] % 10 == 0 or event[3] for event in progress))
            self.assertEqual(model[-1].n_init, 2)
            self.assertTrue(np.isfinite(model[-1].streaming_initialization_scores_[initialization]))
            self.assertTrue(np.isnan(model[-1].streaming_initialization_scores_[1-initialization]))

        split_scores = np.array([model[-1].lower_bound_ for model in split_models])
        np.testing.assert_allclose(
            split_scores, combined[-1].streaming_initialization_scores_, atol=1e-12,
        )
        best = int(np.argmax(split_scores))
        np.testing.assert_allclose(
            split_models[best].score_samples(self.data),
            combined.score_samples(self.data), atol=1e-12,
        )

    def test_invalid_initialization_index_fails_before_reading_batches(self):
        def unreadable():
            raise AssertionError("Invalid indices must fail before reading feature batches")

        for index in (-1, 2, True, 0.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                streaming.fit_streaming_gmm(
                    unreadable, 2, 42, initializations=2, initialization_index=index,
                )

    def test_pipeline_api_finite_and_persistence(self):
        import io

        model = streaming.fit_streaming_gmm(factory(self.data), 2, 19)
        self.assertIsInstance(model[0], StandardScaler)
        self.assertIsInstance(model[1], PCA)
        self.assertIsInstance(model[-1], GaussianMixture)
        self.assertEqual(model.predict(self.data).shape, (len(self.data),))
        probabilities = model.predict_proba(self.data)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(model.score_samples(self.data)).all())
        for covariance in model[-1].covariances_:
            self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_array_equal(model.predict(self.data), restored.predict(self.data))
        np.testing.assert_array_equal(
            model.score_samples(self.data), restored.score_samples(self.data),
        )

    def test_all_rows_each_pass_and_bounded_internal_chunks(self):
        visits = []
        passes = []

        def batches():
            passes.append(len(self.data))
            yield self.data

        original = streaming._transformed_chunks

        def instrumented(*args):
            rows = 0
            for chunk in original(*args):
                self.assertLessEqual(len(chunk), 37)
                rows += len(chunk)
                yield chunk
            visits.append(rows)

        with patch.object(streaming, "_CHUNK_ROWS", 37), patch.object(
            streaming, "_transformed_chunks", instrumented,
        ):
            model = streaming.fit_streaming_gmm(batches, 1, 5, initializations=1, tol=1e-8)
        self.assertEqual(visits, [len(self.data)] * (model[-1].n_iter_ + 2))
        self.assertEqual(len(passes), len(visits) + 1)
        self.assertEqual(model[0].n_samples_seen_, len(self.data))

    def test_read_only_memmap_and_empty_batches(self):
        from pathlib import Path

        path = Path(__file__).parent / "streaming_gmm_test_mmap.npy"
        self.assertFalse(path.exists())
        try:
            np.save(path, self.data.astype(np.float32))
            data = np.load(path, mmap_mode="r")

            def batches():
                yield np.empty((0, 10))
                yield data
                yield np.empty((0, 10))

            with patch.object(streaming, "_CHUNK_ROWS", 43):
                model = streaming.fit_streaming_gmm(batches, 1, 3, initializations=1)
            dense = StandardScaler().fit(data)
            np.testing.assert_allclose(model[0].mean_, dense.mean_, atol=1e-14)
            np.testing.assert_allclose(model[0].var_, dense.var_, atol=1e-13)
            self.assertTrue(np.isfinite(model.score_samples(data)).all())
            self.assertFalse(data.flags.writeable)
        finally:
            if path.exists():
                path.unlink()

    def test_single_component_analytic_solution(self):
        model = streaming.fit_streaming_gmm(factory(self.data, 39), 1, 10, tol=1e-10)
        transformed = model[1].transform(model[0].transform(self.data))
        expected = np.cov(transformed, rowvar=False, bias=True)
        expected.flat[::len(expected) + 1] += 1e-6
        np.testing.assert_allclose(model[-1].means_[0], transformed.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(model[-1].covariances_[0], expected, atol=1e-13)

    def test_invalid_arguments_and_data(self):
        for argument, value in (
            ("components", 0), ("components", True), ("components", 1.5),
            ("initializations", 0), ("max_iter", -1),
            ("tol", 0), ("tol", np.nan), ("tol", np.inf),
        ):
            parameters = {"components": 2, "seed": 0}
            parameters[argument] = value
            with self.subTest(argument=argument, value=value), self.assertRaises(ValueError):
                streaming.fit_streaming_gmm(factory(self.data), **parameters)
        with self.assertRaises(TypeError):
            streaming.fit_streaming_gmm([self.data], 2, 0)
        for data in (
            np.empty((0, 10)), np.ones((1, 10)), np.ones((5, 10)),
            np.ones((5, 9)), np.ones(10), np.full((5, 10), np.nan),
            np.full((5, 10), np.inf), np.ones((5, 10), dtype=complex),
            np.full((5, 10), "bad"),
        ):
            with self.subTest(shape=data.shape, dtype=data.dtype), self.assertRaises(ValueError):
                streaming.fit_streaming_gmm(factory(data), 2, 0)
        with self.assertRaisesRegex(ValueError, "components cannot exceed"):
            streaming.fit_streaming_gmm(factory(self.data[:3]), 4, 0)
        with self.assertRaisesRegex(ValueError, "two training rows"):
            streaming.fit_streaming_gmm(lambda: iter(()), 1, 0)

    def test_changed_or_exhausted_factory(self):
        iterator = iter([self.data])
        with self.assertRaisesRegex(ValueError, "replay the same rows"):
            streaming.fit_streaming_gmm(lambda: iterator, 2, 0)
        calls = 0

        def changing():
            nonlocal calls
            calls += 1
            yield self.data if calls == 1 else self.data[:-1]

        with self.assertRaisesRegex(ValueError, "replay the same rows"):
            streaming.fit_streaming_gmm(changing, 2, 0)

    def test_collapse_and_non_convergence_are_explicit(self):
        model = GaussianMixture(n_components=2)
        with self.assertRaisesRegex(RuntimeError, "collapsed"):
            streaming._maximization(
                model, np.array([100.0, 0.0]), np.zeros((2, 3)), np.zeros((2, 3, 3)), 100,
            )
        with self.assertRaisesRegex(RuntimeError, "factorization failed"):
            streaming._set_parameters(
                model, np.ones(2) / 2, np.zeros((2, 3)),
                np.repeat((-np.eye(3))[None, :, :], 2, axis=0),
            )
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            streaming._set_parameters(
                model, np.array([np.nan, 0.5]), np.zeros((2, 3)),
                np.repeat(np.eye(3)[None, :, :], 2, axis=0),
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(RuntimeError, "No streaming GMM initialization converged"):
                streaming.fit_streaming_gmm(factory(self.data), 2, 42, max_iter=1, tol=1e-15)
        self.assertEqual(len(caught), 2)
        self.assertTrue(all("did not converge" in str(item.message) for item in caught))

    def test_failed_start_reported_when_another_succeeds(self):
        original = streaming._maximization
        calls = 0

        def fail_first(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("component collapsed")
            return original(*args)

        with patch.object(streaming, "_maximization", fail_first), self.assertWarnsRegex(
            RuntimeWarning, "start 1 failed",
        ):
            model = streaming.fit_streaming_gmm(factory(self.data), 1, 42)
        self.assertTrue(model[-1].converged_)
        self.assertTrue(np.isnan(model[-1].streaming_initialization_scores_[0]))
        self.assertEqual(len(model[-1].streaming_initialization_failures_), 1)

    def test_parallel_full_fits_match_serial_and_are_deterministic(self):
        parameters = dict(
            components=2, seed=42, initializations=2, tol=1e-7, chunk_rows=97,
        )
        serial = streaming.fit_streaming_gmm(factory(self.data, 211), **parameters)
        parallel = streaming.fit_streaming_gmm(
            factory(self.data, 211), workers=2, **parameters,
        )
        repeated = streaming.fit_streaming_gmm(
            factory(self.data, 211), workers=2, **parameters,
        )
        for other in (parallel, repeated):
            self.assertEqual(serial[-1].n_iter_, other[-1].n_iter_)
            for attribute in (
                "weights_", "means_", "covariances_", "lower_bounds_",
                "streaming_initialization_scores_",
            ):
                np.testing.assert_array_equal(
                    getattr(serial[-1], attribute), getattr(other[-1], attribute),
                )
            np.testing.assert_array_equal(
                serial.score_samples(self.data), other.score_samples(self.data),
            )

    def test_cached_preprocessing_mmap_and_independent_starts(self):
        preprocessor = streaming.prepare_streaming_preprocessing(
            factory(self.data, 103), chunk_rows=67,
        )
        count, scaler, pca = preprocessor
        self.assertEqual(count, len(self.data))
        transformed = pca.transform(scaler.transform(self.data))

        def unreadable():
            raise AssertionError("Cached fitting must not read any raw features")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "transformed.npy"
            np.save(path, transformed)
            mmap = np.load(path, mmap_mode="r")
            self.assertFalse(mmap.flags.writeable)
            parameters = dict(components=2, seed=42, tol=1e-7, chunk_rows=67)
            raw = streaming.fit_streaming_gmm(
                factory(self.data, 103), preprocessor=preprocessor, **parameters,
            )
            with patch.object(streaming, "_stream_moments", side_effect=AssertionError), \
                    patch.object(scaler, "transform", side_effect=AssertionError), \
                    patch.object(pca, "transform", side_effect=AssertionError):
                cached = []
                for start in range(2):
                    cached.append(streaming.fit_streaming_gmm(
                        unreadable, initialization_index=start,
                        preprocessor=preprocessor, transformed_batches=factory(mmap, 103),
                        workers=2 if start == 0 else 1, **parameters,
                    ))
            for start, model in enumerate(cached):
                self.assertAlmostEqual(
                    model[-1].lower_bound_,
                    raw[-1].streaming_initialization_scores_[start], places=13,
                )
                start_seed = np.random.RandomState(42).randint(
                    np.iinfo(np.int32).max, size=2,
                )[start]
                raw_centers = streaming._initial_centers(
                    factory(self.data, 103), scaler, pca, count, 2,
                    np.random.RandomState(start_seed), 67,
                )
                cached_centers = streaming._initial_centers(
                    unreadable, scaler, pca, count, 2,
                    np.random.RandomState(start_seed), 67, factory(mmap, 103),
                )
                np.testing.assert_array_equal(raw_centers, cached_centers)
            best = max(cached, key=lambda model: model[-1].lower_bound_)
            np.testing.assert_allclose(
                raw.score_samples(self.data), best.score_samples(self.data), atol=1e-12,
            )
            del mmap

    def test_locally_absent_components_are_merged_before_validation(self):
        _, scaler, pca = streaming.prepare_streaming_preprocessing(factory(self.data))
        dimensions = pca.n_components_
        chunks = [
            np.full((4, dimensions), -1000.0),
            np.full((4, dimensions), 1000.0),
        ]
        model = GaussianMixture(n_components=2)
        streaming._set_parameters(
            model, np.array([0.5, 0.5]),
            np.array([chunks[0][0], chunks[1][0]]),
            np.repeat(np.eye(dimensions)[None, :, :], 2, axis=0),
        )
        np.testing.assert_array_equal(
            streaming._chunk_statistics(chunks[0], model)[0], [4.0, 0.0],
        )
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn"),
            initializer=streaming._worker_initialize,
        ) as executor:
            results = streaming._expectation_statistics(
                factory(self.data), scaler, pca, model, 8,
                chunk_rows=3, transformed_batches=lambda: iter(chunks),
                executor=executor, workers=2,
            )
        np.testing.assert_array_equal(results[0], [4.0, 4.0])
        np.testing.assert_array_equal(results[1], model.means_)
        np.testing.assert_array_equal(results[2], 0.0)
        streaming._maximization(model, *results[:3], 8)
        np.testing.assert_array_equal(model.weights_, [0.5, 0.5])

    def test_submission_queue_and_worker_chunks_are_bounded(self):
        class Future:
            def __init__(self, owner, function, args):
                self.owner, self.function, self.args = owner, function, args

            def result(self):
                self.owner.active -= 1
                return self.function(*self.args)

        class Executor:
            active = 0
            maximum = 0
            sizes = []

            def submit(self, function, *args):
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                self.sizes.append(len(args[0]))
                self_outer.assertTrue(args[0].flags.owndata)
                return Future(self, function, args)

        self_outer = self
        executor = Executor()
        model = GaussianMixture(n_components=1)
        streaming._set_parameters(
            model, np.ones(1), np.zeros((1, 10)), np.eye(10)[None, :, :],
        )
        produced = 0
        consumed = 0

        def chunks():
            nonlocal produced
            for chunk in streaming._chunks(lambda: iter([self.data]), 37):
                produced += 1
                self.assertLessEqual(produced - consumed, 3)
                yield chunk

        for _ in streaming._bounded_statistics(chunks(), model, executor, 3):
            consumed += 1
        self.assertEqual(executor.maximum, 3)
        self.assertEqual(executor.active, 0)
        self.assertEqual(sum(executor.sizes), len(self.data))
        self.assertLessEqual(max(executor.sizes), 37)

    def test_weighted_centered_precision_with_large_offsets(self):
        data = 1e10 + self.data
        model = GaussianMixture(n_components=2)
        streaming._set_parameters(
            model, np.array([0.4, 0.6]), data[[20, 300]],
            np.repeat((np.eye(10) * 100)[None, :, :], 2, axis=0),
        )
        mass, mean, scatter, _ = streaming._chunk_statistics(data, model)
        _, log_resp = model._estimate_log_prob_resp(data)
        weights = np.exp(log_resp)
        shifted = data - data[0]
        expected_mean = data[0] + weights.T @ shifted / weights.sum(axis=0)[:, None]
        expected_scatter = np.array([
            (data - expected_mean[k]).T
            @ ((data - expected_mean[k]) * weights[:, k, None]) for k in range(2)
        ])
        np.testing.assert_allclose(mass, weights.sum(axis=0), rtol=1e-14)
        np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=2e-6)
        np.testing.assert_allclose(scatter, expected_scatter, rtol=1e-12, atol=1e-8)

    def test_new_arguments_and_cached_metadata_validation_before_reads(self):
        def unreadable():
            raise AssertionError("Invalid configuration must fail before reading data")

        for name in ("workers", "chunk_rows"):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    streaming.fit_streaming_gmm(unreadable, 2, 42, **{name: value})
        with self.assertRaisesRegex(ValueError, "requires preprocessor"):
            streaming.fit_streaming_gmm(unreadable, 2, 42, transformed_batches=unreadable)
        preprocessor = streaming.prepare_streaming_preprocessing(factory(self.data))
        with self.assertRaisesRegex(TypeError, "transformed_batches"):
            streaming.fit_streaming_gmm(
                unreadable, 2, 42, preprocessor=preprocessor, transformed_batches=[],
            )
        count, scaler, pca = preprocessor
        for invalid in ((), (count + 1, scaler, pca), (True, scaler, pca),
                        (count, None, pca), (count, scaler, None)):
            with self.subTest(invalid_count=invalid[:1]), self.assertRaises(ValueError):
                streaming.fit_streaming_gmm(unreadable, 2, 42, preprocessor=invalid)
        with patch.object(pca, "components_", pca.components_.astype(np.float32)):
            with self.assertRaisesRegex(ValueError, "finite float64"):
                streaming.fit_streaming_gmm(unreadable, 2, 42, preprocessor=preprocessor)
        with patch.object(pca, "whiten", True):
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                streaming.fit_streaming_gmm(unreadable, 2, 42, preprocessor=preprocessor)
        for transformed in (
            np.zeros((count, pca.n_components_ + 1)),
            np.zeros((count - 1, pca.n_components_)),
            np.full((count, pca.n_components_), np.nan),
        ):
            with self.subTest(shape=transformed.shape), self.assertRaises(ValueError):
                streaming.fit_streaming_gmm(
                    unreadable, 2, 42, preprocessor=preprocessor,
                    transformed_batches=factory(transformed),
                )

    def test_parallel_cleanup_on_factory_failure_and_nonconvergence(self):
        children = {child.pid for child in multiprocessing.active_children()}
        preprocessor = streaming.prepare_streaming_preprocessing(factory(self.data))
        _, scaler, pca = preprocessor
        transformed = pca.transform(scaler.transform(self.data))
        calls = 0

        def changing():
            nonlocal calls
            calls += 1
            yield transformed if calls == 1 else transformed[:-1]

        with self.assertRaisesRegex(ValueError, "replay the same rows"):
            streaming.fit_streaming_gmm(
                factory(self.data), 2, 42, workers=2, chunk_rows=97,
                preprocessor=preprocessor, transformed_batches=changing,
            )
        self.assertEqual(children, {child.pid for child in multiprocessing.active_children()})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(RuntimeError, "No streaming GMM"):
                streaming.fit_streaming_gmm(
                    factory(self.data), 2, 42, workers=2, chunk_rows=97,
                    max_iter=1, tol=1e-15,
                )
        self.assertEqual(len(caught), 2)
        self.assertEqual(children, {child.pid for child in multiprocessing.active_children()})

    def test_worker_exceptions_are_explicit_and_pool_is_closed(self):
        children = {child.pid for child in multiprocessing.active_children()}
        model = GaussianMixture(n_components=1)
        streaming._set_parameters(
            model, np.ones(1), np.zeros((1, 10)), np.eye(10)[None, :, :],
        )
        model.weights_[0] = np.nan
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn"),
            initializer=streaming._worker_initialize,
        ) as executor:
            with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                list(streaming._bounded_statistics(
                    streaming._chunks(factory(self.data), 37), model, executor, 2,
                ))
        self.assertEqual(children, {child.pid for child in multiprocessing.active_children()})


if __name__ == "__main__":
    unittest.main()
