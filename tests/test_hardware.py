"""Tests for core.hardware detection and adaptive tuning."""

from __future__ import annotations

import ast
from pathlib import Path

from core.hardware import (
    HardwareProfile,
    blas_threads_per_worker,
    detect_hardware,
    executor_max_workers,
    held_corpora_limit,
    io_worker_threads,
    onnx_execution_providers,
    onnx_tuning,
    recommended_worker_threads,
    reset_cache,
    spacy_pipe_batch_size,
)


def _profile(  # noqa: PLR0913 -- test builder for synthetic hardware profiles
    *,
    logical: int = 8,
    physical: int = 8,
    total_mb: int = 16384,
    avail_mb: int = 8192,
    is_low_memory: bool = False,
    cgroup_cpus: int | None = None,
    cgroup_ram_mb: int | None = None,
) -> HardwareProfile:
    return HardwareProfile(
        logical_cpus=logical,
        physical_cpus=physical,
        total_ram_mb=total_mb,
        available_ram_mb=avail_mb,
        system="Linux",
        machine="x86_64",
        is_low_memory=is_low_memory,
        cgroup_cpu_limit=cgroup_cpus,
        cgroup_ram_mb=cgroup_ram_mb,
    )


def test_detect_hardware_returns_sane_profile_and_caches() -> None:
    reset_cache()
    first = detect_hardware()
    second = detect_hardware()
    assert first is second
    assert first.logical_cpus >= 1
    assert 1 <= first.physical_cpus <= first.logical_cpus
    assert first.total_ram_mb >= 512
    assert first.available_ram_mb >= 128


def test_hardware_module_has_no_import_time_probes() -> None:
    """R1: importing core.hardware must not call detect_hardware or load third-party deps."""
    path = Path(__file__).resolve().parents[1] / "core" / "hardware.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_calls = [
        node.value.func.id
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
    ]
    assert top_calls == []


def test_worker_counts_scale_with_hardware_and_memory() -> None:
    low = _profile(logical=4, physical=2, total_mb=4096, avail_mb=1024, is_low_memory=True)
    mid = _profile(logical=8, physical=4, total_mb=8192, avail_mb=4096)
    high = _profile(logical=32, physical=16, total_mb=65536, avail_mb=32768)

    assert recommended_worker_threads(task_count=10, profile=low) <= 2
    assert recommended_worker_threads(task_count=10, profile=mid) == 4
    assert recommended_worker_threads(task_count=10, profile=high) == 8
    assert recommended_worker_threads(task_count=3, profile=high) == 3

    assert io_worker_threads(2, profile=high) == 1
    assert io_worker_threads(12, profile=high) == 8
    assert executor_max_workers(6, profile=high) == 6
    assert blas_threads_per_worker(4, profile=high) == 4


def test_container_cgroup_limits_cap_detected_resources() -> None:
    container = _profile(logical=64, physical=32, total_mb=131072, avail_mb=65536, cgroup_cpus=2, cgroup_ram_mb=2048)
    assert container.effective_cpus == 2
    assert container.effective_ram_mb == 2048
    assert held_corpora_limit(container) == 1
    assert spacy_pipe_batch_size(container) == 16
    assert onnx_tuning(container).intra_op_threads == 2


def test_ram_tiers_adapt_caches_batches_and_onnx() -> None:
    small = _profile(physical=2, total_mb=4096, avail_mb=1500, is_low_memory=True)
    medium = _profile(physical=6, total_mb=8192, avail_mb=4096)
    large = _profile(physical=16, total_mb=32768, avail_mb=16384)

    assert held_corpora_limit(small) == 1
    assert held_corpora_limit(medium) == 2
    assert held_corpora_limit(large) == 4

    assert spacy_pipe_batch_size(small) == 16
    assert spacy_pipe_batch_size(medium) == 32
    assert spacy_pipe_batch_size(large) == 64

    small_onnx = onnx_tuning(small)
    large_onnx = onnx_tuning(large)
    assert small_onnx.token_budget < large_onnx.token_budget
    assert small_onnx.max_batch < large_onnx.max_batch
    assert large_onnx.intra_op_threads == 8


def test_onnx_provider_selection_excludes_cloud_and_prefers_local() -> None:
    assert onnx_execution_providers(["AzureExecutionProvider", "CPUExecutionProvider"]) == ("CPUExecutionProvider",)
    assert onnx_execution_providers(["CUDAExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]) == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert onnx_execution_providers(["CoreMLExecutionProvider"]) == (
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )
