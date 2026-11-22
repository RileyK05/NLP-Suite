"""Hardware detection and adaptive runtime tuning.

**R1 (no import-time side effects):** importing this module reads zero hardware
probes, environment state, or native extensions. All detection happens lazily
when :func:`detect_hardware` (or one of the tuning helpers) is called.

Every tuning function accepts an explicit ``profile`` override so callers and
tests can evaluate low-end, mid-range, or workstation profiles deterministically
without mutating process state.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import importlib
import importlib.util
import os
from pathlib import Path
import platform
import sys

__all__ = [
    "HardwareProfile",
    "OnnxTuning",
    "blas_threads_per_worker",
    "detect_hardware",
    "executor_max_workers",
    "held_corpora_limit",
    "io_worker_threads",
    "onnx_execution_providers",
    "onnx_tuning",
    "recommended_worker_threads",
    "reset_cache",
    "spacy_pipe_batch_size",
]

_MIB = 1024 * 1024
_DEFAULT_RAM_MB = 4096
_TINY_RAM_MB = 4096
_LOW_RAM_MB = 6144
_MED_RAM_MB = 14336
_LOW_AVAIL_RAM_MB = 1536
_MIN_ONNX_AVAIL_MB = 2048
_MIN_PARALLEL_IO_FILES = 4
_MIN_SMT_LOGICAL_CPUS = 4
_MAX_POOL_WORKERS = 8

# Local execution providers that accelerate inference deterministically on device.
# Cloud/remote providers (e.g. AzureExecutionProvider) are intentionally excluded.
# CoreML is excluded too: the macOS onnxruntime wheel lists it, the session
# opens, and then every prediction on the suite's transformer models fails
# ("Unable to compute the prediction"), past the session-open CPU fallback in
# core/models/onnx_backend.py. CUDA and ROCm appear only with a GPU build of
# onnxruntime that a user installs on purpose; the installers never ship one.
_PREFERRED_LOCAL_EPS: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
)


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Snapshot of the host CPU, memory, OS, and optional accelerator capabilities."""

    logical_cpus: int
    physical_cpus: int
    total_ram_mb: int
    available_ram_mb: int
    system: str
    machine: str
    is_low_memory: bool
    cgroup_cpu_limit: int | None = None
    cgroup_ram_mb: int | None = None

    @property
    def effective_cpus(self) -> int:
        if self.cgroup_cpu_limit is not None:
            return max(1, min(self.physical_cpus, self.cgroup_cpu_limit))
        return max(1, self.physical_cpus)

    @property
    def effective_ram_mb(self) -> int:
        if self.cgroup_ram_mb is not None:
            return max(512, min(self.total_ram_mb, self.cgroup_ram_mb))
        return max(512, self.total_ram_mb)


@dataclass(frozen=True, slots=True)
class OnnxTuning:
    """Adaptive ONNX Runtime session and batching configuration."""

    intra_op_threads: int
    inter_op_threads: int
    token_budget: int
    max_batch: int
    enable_mem_pattern: bool


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _windows_ram_mb() -> tuple[int, int] | None:
    # if/else on the platform, not an early return: mypy reads sys.platform
    # statically and skips the other branch silently, but code after an early
    # return (or after a branch that always returns) is "unreachable" on one
    # platform, which warn_unreachable reports. CI type-checks all three.
    if sys.platform == "win32":

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            kernel32 = getattr(ctypes, "windll", None)
            if kernel32 is None or not kernel32.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return max(512, int(status.ullTotalPhys // _MIB)), max(128, int(status.ullAvailPhys // _MIB))
        except OSError:
            return None
    else:
        return None


def _proc_meminfo_mb() -> tuple[int, int] | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    total_kb: int | None = None
    avail_kb: int | None = None
    try:
        for line in meminfo.read_text(encoding="ascii", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail_kb = int(line.split()[1])
            if total_kb is not None and avail_kb is not None:
                break
    except (OSError, ValueError, IndexError):
        return None
    if total_kb is None:
        return None
    total_mb = max(512, total_kb // 1024)
    avail_mb = max(128, (avail_kb // 1024) if avail_kb is not None else total_mb // 2)
    return total_mb, avail_mb


def _posix_sysconf_ram_mb() -> tuple[int, int] | None:
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    sysconf_names: dict[str, int] = getattr(os, "sysconf_names", {})
    try:
        pages = int(sysconf("SC_PHYS_PAGES"))
        page_size = int(sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            total_mb = max(512, (pages * page_size) // _MIB)
            avail_pages = int(sysconf("SC_AVPHYS_PAGES")) if "SC_AVPHYS_PAGES" in sysconf_names else -1
            avail_mb = max(128, (avail_pages * page_size) // _MIB) if avail_pages > 0 else total_mb // 2
            return total_mb, avail_mb
    except (OSError, ValueError):
        return None
    return None


def _detect_ram_mb() -> tuple[int, int]:
    psutil = sys.modules.get("psutil")
    if psutil is not None:
        with suppress(Exception):
            vm = psutil.virtual_memory()
            return max(512, int(vm.total // _MIB)), max(128, int(vm.available // _MIB))
    for probe in (_windows_ram_mb, _proc_meminfo_mb, _posix_sysconf_ram_mb):
        result = probe()
        if result is not None:
            return result
    return _DEFAULT_RAM_MB, _DEFAULT_RAM_MB // 2


def _cgroup_limits() -> tuple[int | None, int | None]:
    """Read Linux cgroup v2 / v1 CPU and memory quotas when running inside containers."""
    current_platform: str = sys.platform
    if current_platform != "linux":
        return None, None
    cpu_limit: int | None = None
    ram_limit: int | None = None

    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        with suppress(OSError, ValueError, IndexError):
            quota_s, period_s = cpu_max.read_text(encoding="ascii").strip().split()[:2]
            if quota_s != "max":
                quota, period = int(quota_s), int(period_s)
                if quota > 0 and period > 0:
                    cpu_limit = max(1, round(quota / period))

    mem_max = Path("/sys/fs/cgroup/memory.max")
    if mem_max.is_file():
        with suppress(OSError, ValueError):
            raw = mem_max.read_text(encoding="ascii").strip()
            if raw != "max":
                ram_limit = max(512, int(raw) // _MIB)
    elif (v1_mem := Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")).is_file():
        with suppress(OSError, ValueError):
            value = int(v1_mem.read_text(encoding="ascii").strip())
            # Ignore kernel sentinel (~2^63) meaning unlimited.
            if 0 < value < (1 << 50):
                ram_limit = max(512, value // _MIB)
    return cpu_limit, ram_limit


def _linux_physical_cpus(logical: int) -> int | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None
    try:
        pairs: set[tuple[str, str]] = set()
        phys_id = "0"
        core_id: str | None = None
        for line in cpuinfo.read_text(encoding="ascii", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped:
                if core_id is not None:
                    pairs.add((phys_id, core_id))
                phys_id, core_id = "0", None
                continue
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                key, val = key.strip(), val.strip()
                if key == "physical id":
                    phys_id = val
                elif key == "core id":
                    core_id = val
        if core_id is not None:
            pairs.add((phys_id, core_id))
        if pairs:
            return min(logical, len(pairs))
    except OSError:
        return None
    return None


def _detect_physical_cpus(logical: int) -> int:
    psutil = sys.modules.get("psutil")
    if psutil is not None:
        with suppress(Exception):
            physical = psutil.cpu_count(logical=False)
            if isinstance(physical, int) and physical > 0:
                return min(logical, physical)
    if sys.platform == "linux":
        linux_cores = _linux_physical_cpus(logical)
        if linux_cores is not None:
            return linux_cores
    if sys.platform == "win32" and importlib.util.find_spec("psutil") is not None:
        with suppress(Exception):
            psutil_mod = importlib.import_module("psutil")
            physical = psutil_mod.cpu_count(logical=False)
            if isinstance(physical, int) and physical > 0:
                return min(logical, physical)
    # Assume typical SMT on x86_64 when topology is unavailable.
    if logical >= _MIN_SMT_LOGICAL_CPUS and platform.machine().lower() in {"amd64", "x86_64"}:
        return max(1, logical // 2)
    return max(1, logical)


@lru_cache(maxsize=1)
def _cached_profile() -> HardwareProfile:
    logical = max(1, os.cpu_count() or 1)
    physical = _detect_physical_cpus(logical)
    total_ram_mb, available_ram_mb = _detect_ram_mb()
    cgroup_cpus, cgroup_ram = _cgroup_limits()
    effective_ram = min(total_ram_mb, cgroup_ram) if cgroup_ram is not None else total_ram_mb
    return HardwareProfile(
        logical_cpus=logical,
        physical_cpus=physical,
        total_ram_mb=total_ram_mb,
        available_ram_mb=min(available_ram_mb, effective_ram),
        system=platform.system() or sys.platform,
        machine=platform.machine() or "",
        is_low_memory=effective_ram <= _LOW_RAM_MB or available_ram_mb <= _LOW_AVAIL_RAM_MB,
        cgroup_cpu_limit=cgroup_cpus,
        cgroup_ram_mb=cgroup_ram,
    )


def detect_hardware(*, force_refresh: bool = False) -> HardwareProfile:
    """Return the detected host hardware profile (cached per process)."""
    if force_refresh:
        _cached_profile.cache_clear()
    return _cached_profile()


def reset_cache() -> None:
    """Clear the cached hardware profile (intended for unit tests)."""
    _cached_profile.cache_clear()


def _profile_or_detect(profile: HardwareProfile | None) -> HardwareProfile:
    return profile if profile is not None else detect_hardware()


def recommended_worker_threads(
    task_count: int = 0,
    profile: HardwareProfile | None = None,
    *,
    memory_per_worker_mb: int = 128,
) -> int:
    """Thread pool size for pure CPU/Python analysis tasks.

    Honors ``NLP_MAX_WORKERS`` when set, respects cgroup quotas, and scales
    down on low-memory systems so concurrent frames do not push into swap.
    """
    override = _positive_int_env("NLP_MAX_WORKERS")
    hw = _profile_or_detect(profile)
    cap = hw.effective_cpus
    if memory_per_worker_mb > 0:
        mem_cap = max(1, hw.available_ram_mb // max(64, memory_per_worker_mb))
        cap = min(cap, mem_cap)
    workers = max(1, min(cap, _MAX_POOL_WORKERS))
    if override is not None:
        workers = max(1, min(override, hw.logical_cpus * 2))
    if hw.is_low_memory and override is None:
        workers = min(workers, 2)
    if task_count > 0:
        workers = min(workers, task_count)
    return max(1, workers)


def executor_max_workers(task_count: int = 0, profile: HardwareProfile | None = None) -> int:
    """Bounded worker count for concurrent pure profiler tool execution."""
    return recommended_worker_threads(task_count=task_count, profile=profile, memory_per_worker_mb=192)


def io_worker_threads(item_count: int = 0, profile: HardwareProfile | None = None) -> int:
    """Worker count for parallel file reads; stays single-threaded on tiny batches."""
    if 0 < item_count < _MIN_PARALLEL_IO_FILES:
        return 1
    hw = _profile_or_detect(profile)
    workers = max(1, min(hw.logical_cpus, _MAX_POOL_WORKERS))
    if item_count > 0:
        workers = min(workers, item_count)
    return workers


def held_corpora_limit(profile: HardwareProfile | None = None) -> int:
    """Number of parsed workspace corpora to keep resident in memory."""
    hw = _profile_or_detect(profile)
    ram = hw.effective_ram_mb
    if ram < _LOW_RAM_MB:
        return 1
    if ram < _MED_RAM_MB:
        return 2
    return 4


def spacy_pipe_batch_size(profile: HardwareProfile | None = None) -> int:
    """Adaptive ``nlp.pipe(..., batch_size=...)`` sized to host memory."""
    hw = _profile_or_detect(profile)
    if hw.is_low_memory or hw.effective_ram_mb < _LOW_RAM_MB:
        return 16
    if hw.effective_ram_mb < _MED_RAM_MB:
        return 32
    return 64


def onnx_tuning(profile: HardwareProfile | None = None) -> OnnxTuning:
    """Compute ONNX Runtime session thread counts and dynamic batching caps."""
    hw = _profile_or_detect(profile)
    omp = _positive_int_env("OMP_NUM_THREADS")
    cpus = hw.effective_cpus
    intra = min(omp, hw.logical_cpus) if omp is not None else max(1, min(cpus, _MAX_POOL_WORKERS))
    if hw.available_ram_mb < _MIN_ONNX_AVAIL_MB or hw.effective_ram_mb < _TINY_RAM_MB:
        return OnnxTuning(
            intra_op_threads=intra,
            inter_op_threads=1,
            token_budget=2048,
            max_batch=32,
            enable_mem_pattern=True,
        )
    if hw.effective_ram_mb < _MED_RAM_MB:
        return OnnxTuning(
            intra_op_threads=intra,
            inter_op_threads=1,
            token_budget=4096,
            max_batch=64,
            enable_mem_pattern=True,
        )
    return OnnxTuning(
        intra_op_threads=intra,
        inter_op_threads=1,
        token_budget=8192,
        max_batch=96,
        enable_mem_pattern=True,
    )


def onnx_execution_providers(available_providers: Sequence[str] | None = None) -> tuple[str, ...]:
    """Select verified local execution providers ordered from fastest to CPU fallback."""
    if available_providers is None:
        if "onnxruntime" in sys.modules:
            available_providers = sys.modules["onnxruntime"].get_available_providers()
        elif importlib.util.find_spec("onnxruntime") is not None:
            ort = importlib.import_module("onnxruntime")
            available_providers = ort.get_available_providers()
        else:
            available_providers = ()
    available_set = set(available_providers)
    chosen = [ep for ep in _PREFERRED_LOCAL_EPS if ep in available_set]
    chosen.append("CPUExecutionProvider")
    return tuple(chosen)


def blas_threads_per_worker(workers: int, profile: HardwareProfile | None = None) -> int:
    """Divide BLAS thread budget across *workers* so matrix ops do not oversubscribe CPUs."""
    hw = _profile_or_detect(profile)
    return max(1, hw.effective_cpus // max(1, workers))
