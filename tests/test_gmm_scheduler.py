"""Shared CPU allocation, cleanup and deterministic complete-fit regressions."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import ExitStack
import os
import threading

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from gmm_scheduler import GMMStatisticsScheduler, SweepProgress, run_gmm_jobs
from streaming_gmm import (
    MappedTrainingData, ProgressCallbackError, fit_streaming_gmm,
    prepare_streaming_preprocessing,
)


@pytest.mark.parametrize("cpus", [1, 2, 5, 49])
def test_budget_and_tail_allocation(cpus):
    scheduler = GMMStatisticsScheduler(cpus, 5)
    slots = max(1, cpus - 1)
    with scheduler.iteration() as (_, workers):
        assert workers == min(32, max(1, slots // 5))
        assert scheduler.leased <= slots
    assert scheduler.leased == 0
    for _ in range(4):
        scheduler.finish()
    with scheduler.iteration() as (_, workers):
        assert workers == min(32, slots)
    assert scheduler.maximum_leased <= slots


def test_concurrent_leases_do_not_oversubscribe():
    scheduler = GMMStatisticsScheduler(5, 4)
    with ExitStack() as stack:
        for _ in range(4):
            _, workers = stack.enter_context(scheduler.iteration())
            assert workers == 1
        assert scheduler.leased == 4
    assert scheduler.leased == 0


def test_exception_releases_allocation_and_cancel_wakes_waiter():
    scheduler = GMMStatisticsScheduler(2, 2)
    with pytest.raises(ValueError, match="injected"):
        with scheduler.iteration():
            raise ValueError("injected")
    assert scheduler.leased == 0
    entered = threading.Event()

    def waiting():
        entered.set()
        with scheduler.iteration():
            raise AssertionError("A cancelled waiter must not run")

    with scheduler.iteration(), ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(waiting)
        assert entered.wait(timeout=5)
        scheduler.cancel()
        with pytest.raises(CancelledError):
            future.result(timeout=5)
    assert scheduler.leased == 0


@pytest.mark.parametrize("cpus", [0, True, 1.5])
def test_invalid_allocation(cpus):
    with pytest.raises(ValueError, match="cpus"):
        run_gmm_jobs([1], lambda *args: None, cpus)


def test_queue_and_error_propagation():
    results = []
    run_gmm_jobs([3, 2, 1], lambda job, scheduler: results.append(job), 1)
    assert results == [3, 2, 1]

    def fail(job, scheduler):
        with scheduler.iteration():
            raise OSError("injected infrastructure failure")

    with pytest.raises(OSError, match="infrastructure"):
        run_gmm_jobs([3, 2, 1], fail, 3)
    run_gmm_jobs([], fail, 1)


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux CPU affinity")
def test_coordinator_affinity_and_restoration():
    original = os.sched_getaffinity(0)
    with GMMStatisticsScheduler(3, 1):
        assert os.sched_getaffinity(0) == {min(original)}
    assert os.sched_getaffinity(0) == original
    with pytest.raises(OSError, match="injected"):
        with GMMStatisticsScheduler(3, 1):
            raise OSError("injected")
    assert os.sched_getaffinity(0) == original


def test_shared_pool_complete_fits_match_serial():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(1400, 3))
    latent[:, 0] += rng.choice([-3., 3.], len(latent))
    data = latent @ rng.normal(size=(3, 10)) + .1 * rng.normal(size=(1400, 10))
    raw = lambda: (data[start:start + 97] for start in range(0, len(data), 97))
    jobs = [(3, 42, 0), (2, 73, 1), (1, 42, 0)]
    with threadpool_limits(limits=1):
        preprocessing = prepare_streaming_preprocessing(raw, chunk_rows=97)
        _, scaler, pca = preprocessing
        transformed = pca.transform(scaler.transform(data))
        batches = lambda: (transformed[start:start + 97]
                           for start in range(0, len(data), 97))

        def fit(job, scheduler=None):
            components, seed, initialization = job
            return fit_streaming_gmm(
                raw, components, seed, initialization_index=initialization,
                chunk_rows=97, preprocessor=preprocessing, transformed_batches=batches,
                statistics_scheduler=scheduler,
            )

        reference = {job: fit(job) for job in jobs}
        results = {}

        def operation(job, scheduler):
            results[job] = fit(job, scheduler)

        run_gmm_jobs(jobs, operation, 5)
        for job in jobs:
            left, right = reference[job][-1], results[job][-1]
            assert left.n_iter_ == right.n_iter_
            for attribute in ("weights_", "means_", "covariances_", "lower_bounds_"):
                np.testing.assert_array_equal(getattr(left, attribute), getattr(right, attribute))
            np.testing.assert_array_equal(reference[job].predict(data), results[job].predict(data))
            np.testing.assert_array_equal(reference[job].score_samples(data),
                                          results[job].score_samples(data))
        with pytest.raises(ValueError, match="requires workers=1"):
            fit_streaming_gmm(raw, 2, 42, workers=2, statistics_scheduler=object())


@pytest.mark.parametrize("cpus", [1, 5])
def test_file_backed_complete_fits_match_array_path(tmp_path, cpus):
    rng = np.random.default_rng(12)
    data = rng.normal(size=(1001, 10))
    data[:, :3] += rng.choice([-2., 2.], size=(1001, 1))
    raw = lambda: (data[start:start + 97] for start in range(0, len(data), 97))
    with threadpool_limits(limits=1):
        preprocessing = prepare_streaming_preprocessing(raw, chunk_rows=37)
        _, scaler, pca = preprocessing
        transformed = pca.transform(scaler.transform(data))
        paths = [tmp_path / "day1.npy", tmp_path / "day2.npy"]
        np.save(paths[0], transformed[:413])
        np.save(paths[1], transformed[413:])
        mapped = MappedTrainingData(paths, 97, pca.n_components_)
        array_batches = lambda: (
            batch for day in (transformed[:413], transformed[413:])
            for batch in (day[start:start + 97] for start in range(0, len(day), 97)))
        jobs = [(2, 42, 0), (3, 73, 1), (1, 109, 0)]
        reference = {}
        for job in jobs:
            reference[job] = fit_streaming_gmm(
                raw, job[0], job[1], initialization_index=job[2],
                preprocessor=preprocessing, transformed_batches=array_batches,
                chunk_rows=37,
            )
        events = []
        results = {}
        def operation(job, scheduler):
            results[job] = fit_streaming_gmm(
                raw, job[0], job[1], initialization_index=job[2],
                preprocessor=preprocessing, mapped_data=mapped, chunk_rows=37,
                statistics_scheduler=scheduler, event_callback=events.append,
            )
        run_gmm_jobs(jobs, operation, cpus)
        for job in jobs:
            left, right = reference[job][-1], results[job][-1]
            assert left.n_iter_ == right.n_iter_
            for attribute in ("weights_", "means_", "covariances_", "lower_bounds_"):
                np.testing.assert_array_equal(getattr(left, attribute), getattr(right, attribute))
            np.testing.assert_array_equal(reference[job].predict(data), results[job].predict(data))
            np.testing.assert_array_equal(reference[job].score_samples(data),
                                          results[job].score_samples(data))
        completed = [event for event in events if event["event"] == "pass_completed"]
        assert all(event["processed_pixels"] == len(data) for event in completed)
        assert all(event["duration_seconds"] > 0 for event in completed)
        assert all(event["pixels_per_second"] > 0 for event in completed)
        assert sum(event["iteration"] == 0 for event in completed) == len(jobs)
        assert sum(event["event"] == "iteration_completed" for event in events) == sum(
            model[-1].n_iter_ for model in results.values())


def test_sweep_progress_concurrent_events_and_restart(tmp_path):
    import json
    jobs = [(k, 42, 0) for k in range(20)]
    with SweepProgress(tmp_path, jobs, reused=2) as progress:
        def operation(job):
            progress.emit({"event": "fit_started", "job": job})
            progress.emit({"event": "pass_completed", "job": job, "processed_pixels": 123})
            progress.emit({"event": "fit_completed", "job": job})
        with ThreadPoolExecutor(max_workers=5) as threads:
            list(threads.map(operation, jobs))
    records = [json.loads(line) for line in progress.path.read_text().splitlines()]
    assert len(records) == 3 * len(jobs) + 2
    assert records[-1]["fits"] == {
        "queued": 0, "running": 0, "completed": 20, "failed": 0, "reused": 2}
    assert records[-1]["completed_pass_pixels"] == 20 * 123
    assert records[-1]["event"] == "sweep_completed"
    assert all(record["schema_version"] == 1 for record in records)
    assert all(record["timestamp_utc"].endswith("+00:00") for record in records)
    assert records == sorted(records, key=lambda record: record["elapsed_seconds"])
    with pytest.raises(OSError, match="interrupted"):
        with SweepProgress(tmp_path, jobs) as restarted:
            raise OSError("interrupted")
    assert restarted.path != progress.path
    assert json.loads(restarted.path.read_text().splitlines()[-1])["event"] == "sweep_failed"


def test_progress_failure_propagates_as_infrastructure_error():
    data = np.random.default_rng(42).normal(size=(200, 10))
    def fail(event):
        raise RuntimeError("logger unavailable")
    with pytest.raises(ProgressCallbackError, match="progress callback") as error:
        fit_streaming_gmm(lambda: iter([data]), 1, 42, event_callback=fail)
    assert str(error.value.__cause__) == "logger unavailable"


def test_sweep_log_write_failure_is_explicit(tmp_path):
    with SweepProgress(tmp_path, [(2, 42, 0)]) as progress:
        handle = progress.handle
        class FailedWriter:
            def write(self, value):
                raise OSError("log filesystem unavailable")
            def flush(self):
                handle.flush()
        progress.handle = FailedWriter()
        try:
            with pytest.raises(OSError, match="log filesystem"):
                progress.emit({"event": "pass_started"})
        finally:
            progress.handle = handle
