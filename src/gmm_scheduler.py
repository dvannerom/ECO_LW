"""CPU-budgeted, shared-pool execution of independent streaming GMM fits."""

from concurrent.futures import CancelledError, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import multiprocessing
from numbers import Integral
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

from threadpoolctl import threadpool_limits

from streaming_gmm import _worker_initialize


class SweepProgress:
    """One locked, flushed JSONL event stream per sweep; O(number of fits) state."""

    def __init__(self, directory, jobs, reused=0):
        self.started = time.monotonic()
        self.last_summary = self.started
        self.invocation = uuid4().hex
        self.states = {tuple(job): "queued" for job in jobs}
        self.reused = reused
        self.processed = 0
        self.lock = threading.Lock()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.path = directory / f"{timestamp}-{self.invocation}.jsonl"
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("x")
        try:
            self.emit({"event": "sweep_started", "pending": len(self.states),
                       "reused": self.reused})
            print(f"GMM progress log: {self.path}", flush=True)
        except (OSError, ValueError, TypeError):
            self.handle.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.emit({"event": "sweep_completed" if exc_type is None else "sweep_failed",
                       "error": None if exc_value is None else str(exc_value)})
        finally:
            self.handle.close()

    def emit(self, event):
        with self.lock:
            now = time.monotonic()
            record = {"schema_version": 1, "invocation": self.invocation,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": now - self.started, **event}
            job = record.get("job")
            if job is not None and record["event"] in (
                    "fit_started", "fit_completed", "fit_failed"):
                self.states[tuple(job)] = {
                    "fit_started": "running", "fit_completed": "completed",
                    "fit_failed": "failed",
                }[record["event"]]
            if record["event"] == "pass_completed":
                self.processed += record["processed_pixels"]
            counts = {state: sum(value == state for value in self.states.values())
                      for state in ("queued", "running", "completed", "failed")}
            record["fits"] = {**counts, "reused": self.reused}
            record["completed_pass_pixels"] = self.processed
            self.handle.write(json.dumps(record, allow_nan=False) + "\n")
            self.handle.flush()
            if now - self.last_summary >= 30:
                print(
                    f"GMM sweep elapsed={now - self.started:.0f}s: "
                    f"{counts['completed']} completed, {counts['running']} running, "
                    f"{counts['queued']} queued, {counts['failed']} failed, "
                    f"{self.reused} reused; completed-pass pixels={self.processed}",
                    flush=True,
                )
                self.last_summary = now


def _initialize_compute_worker(cpus):
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
    _worker_initialize()


class GMMStatisticsScheduler:
    """Lease compute slots for one EM pass, returning results in input order.

    The calling process reserves one CPU for coordination when possible.
    At most 32 slots are assigned to a fit; total leases never exceed the
    remaining allocation. Working memory is bounded by slots times chunk size,
    independently of the training population.
    """

    def __init__(self, cpus, jobs):
        for name, value in (("cpus", cpus), ("jobs", jobs)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.cpus = int(cpus)
        self.slots = max(1, self.cpus - 1)
        self.remaining = int(jobs)
        self.leased = 0
        self.maximum_leased = 0
        self.cancelled = False
        self.condition = threading.Condition()
        self.local = threading.local()
        self.executor = None
        self.original_affinity = None

    def __enter__(self):
        compute_cpus = None
        if hasattr(os, "sched_getaffinity"):
            self.original_affinity = os.sched_getaffinity(0)
            allowed = sorted(self.original_affinity)
            allocation = min(self.cpus, len(allowed))
            self.slots = max(1, allocation - 1)
            if allocation < self.cpus:
                print(f"GMM CPU allocation capped by affinity: {allocation}/{self.cpus}",
                      flush=True)
            compute_cpus = set(allowed[1:allocation])
            os.sched_setaffinity(0, {allowed[0]})
        try:
            if self.slots > 1:
                self.executor = ProcessPoolExecutor(
                    max_workers=self.slots,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_compute_worker, initargs=(compute_cpus,),
                )
            return self
        except BaseException:
            self._restore_affinity()
            raise

    def _restore_affinity(self):
        if self.original_affinity is not None:
            os.sched_setaffinity(0, self.original_affinity)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self._restore_affinity()

    @contextmanager
    def iteration(self):
        with self.condition:
            while self.leased == self.slots and not self.cancelled:
                self.condition.wait()
            if self.cancelled:
                raise CancelledError("GMM sweep cancelled after another fit failed")
            workers = min(32, max(1, self.slots // self.remaining),
                          self.slots - self.leased)
            self.leased += workers
            self.maximum_leased = max(self.maximum_leased, self.leased)
        try:
            if getattr(self.local, "workers", None) != workers:
                print(f"GMM job={getattr(self.local, 'job', None)}: "
                      f"compute workers={workers}, unfinished starts={self.remaining}",
                      flush=True)
                self.local.workers = workers
            yield self.executor, workers
        finally:
            with self.condition:
                self.leased -= workers
                self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.remaining -= 1
            self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.cancelled = True
            self.condition.notify_all()


def run_gmm_jobs(jobs, operation, cpus):
    """Run an ordered queue; operation(job, scheduler) persists each result.

    Numerical failed-start records are handled by operation. Infrastructure
    errors cancel the queue, drain running chunk tasks, and propagate to the
    caller. Completed results remain usable on restart.
    """
    jobs = list(jobs)
    if not jobs:
        return
    with threadpool_limits(limits=1), GMMStatisticsScheduler(cpus, len(jobs)) as scheduler:
        def run(job):
            scheduler.local.job = job
            scheduler.local.workers = None
            try:
                return operation(job, scheduler)
            finally:
                scheduler.finish()

        with ThreadPoolExecutor(max_workers=min(scheduler.slots, len(jobs))) as threads:
            futures = [threads.submit(run, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                scheduler.cancel()
                for future in futures:
                    future.cancel()
                raise
