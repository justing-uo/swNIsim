"""JIT loader and disk-cached wrapper for swNIsim fused CUDA kernels.

Compiles `swnisim_cuda_kernels.cu` once per (source, torch, CUDA, GPU arch, Python)
into `~/.cache/swnisim_cuda/` using `torch.utils.cpp_extension.load` when `ninja`
is installed, or directly via `nvcc -shared` when `ninja` is absent (zero extra
Python dependencies on Colab or lab GPU servers).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from typing import Any, List, Optional

import torch

_CUDA_EXT: Any = None
_CUDA_TRIED: bool = False
_CUDA_ERROR: str = ""
_EXT_NAME = "swnisim_cuda_ext_v3"


def _find_nvcc() -> Optional[str]:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    for cand in (
        os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc"),
        os.path.join(os.environ.get("CUDA_PATH", ""), "bin", "nvcc"),
        "/usr/local/cuda/bin/nvcc",
    ):
        if cand and os.path.isfile(cand):
            return cand
    return None


def _cache_dir(src_text: str) -> str:
    base = os.environ.get(
        "SWNISIM_CUDA_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "swnisim_cuda"),
    )
    cuda_ver = str(getattr(torch.version, "cuda", "none"))
    torch_ver = str(torch.__version__)
    try:
        cap = torch.cuda.get_device_capability()
        arch = f"sm{cap[0]}{cap[1]}"
    except Exception:
        arch = "sm_unknown"
    py_ver = f"py{sys.version_info.major}{sys.version_info.minor}"
    key = hashlib.sha256(
        f"{src_text}|{torch_ver}|{cuda_ver}|{arch}|{py_ver}".encode("utf-8")
    ).hexdigest()[:16]
    d = os.path.join(base, f"swnisim_cuda_v3_{arch}_{key}")
    os.makedirs(d, exist_ok=True)
    return d


def _load_so(so_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(_EXT_NAME, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec for {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compile_direct_nvcc(cu_path: str, build_dir: str, so_path: str) -> Any:
    nvcc = _find_nvcc()
    if not nvcc:
        raise RuntimeError("nvcc not found in PATH or /usr/local/cuda/bin/nvcc")

    from torch.utils import cpp_extension as ce

    try:
        inc_dirs: List[str] = list(ce.include_paths(device_type="cuda"))
    except TypeError:
        inc_dirs = list(ce.include_paths(cuda=True))
    py_inc = sysconfig.get_path("include")
    if py_inc and py_inc not in inc_dirs:
        inc_dirs.append(py_inc)
    plat_inc = sysconfig.get_path("platinclude")
    if plat_inc and plat_inc not in inc_dirs:
        inc_dirs.append(plat_inc)

    try:
        lib_dirs: List[str] = list(ce.library_paths(device_type="cuda"))
    except TypeError:
        lib_dirs = list(ce.library_paths(cuda=True))

    cap = torch.cuda.get_device_capability()
    abi = 1 if bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True)) else 0

    tmp_so = f"{so_path}.tmp.{os.getpid()}"
    errs: List[str] = []
    for std_flag in ("-std=c++20", "-std=c++17"):
        cmd: List[str] = [
            nvcc,
            std_flag,
            "-O3",
            "-Xptxas=-O3",
            "--expt-relaxed-constexpr",
            "-shared",
            "--compiler-options",
            "-fPIC,-O3",
            f"-gencode=arch=compute_{cap[0]}{cap[1]},code=sm_{cap[0]}{cap[1]}",
            f"-DTORCH_EXTENSION_NAME={_EXT_NAME}",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        ]
        for d in inc_dirs:
            if d and os.path.isdir(d):
                cmd.append(f"-I{d}")
        cmd.append(cu_path)
        cmd.extend(["-o", tmp_so])
        for d in lib_dirs:
            if d and os.path.isdir(d):
                cmd.append(f"-L{d}")
                cmd.extend(["-Xlinker", f"-rpath,{d}"])
        cmd.extend(["-lc10", "-lc10_cuda", "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-ltorch_python"])

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode == 0:
            os.replace(tmp_so, so_path)
            return _load_so(so_path)
        if os.path.exists(tmp_so):
            os.remove(tmp_so)
        errs.append(f"nvcc compilation ({std_flag}) failed (rc={proc.returncode}):\n{proc.stdout}")
        # Only retry c++17 if nvcc rejected -std=c++20 itself
        if "c++20" not in proc.stdout.lower() and "unrecognized" not in proc.stdout.lower():
            break
    raise RuntimeError("\n---\n".join(errs))


def get_cuda_ext() -> Optional[Any]:
    """Compile (once, disk-cached) and return the fused CUDA kernel extension, or None."""
    global _CUDA_EXT, _CUDA_TRIED, _CUDA_ERROR
    if _CUDA_TRIED:
        return _CUDA_EXT
    _CUDA_TRIED = True

    if os.environ.get("SWNISIM_DISABLE_CUDA_KERNEL", "0") == "1":
        _CUDA_ERROR = "Disabled via SWNISIM_DISABLE_CUDA_KERNEL=1"
        return None
    if not torch.cuda.is_available():
        _CUDA_ERROR = "CUDA not available"
        return None

    cu_path = os.path.join(os.path.dirname(__file__), "swnisim_cuda_kernels.cu")
    if not os.path.isfile(cu_path):
        _CUDA_ERROR = f"Missing CUDA kernel source: {cu_path}"
        return None

    try:
        with open(cu_path, "r", encoding="utf-8") as fh:
            cu_src = fh.read()
        build_dir = _cache_dir(cu_src)
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        so_path = os.path.join(build_dir, f"{_EXT_NAME}{ext_suffix}")

        # Fast path: already compiled and cached on disk
        if os.path.isfile(so_path):
            _CUDA_EXT = _load_so(so_path)
            _CUDA_ERROR = ""
            return _CUDA_EXT

        # File lock prevents concurrent multi-GPU workers (e.g. 2x L40S) from
        # racing on JIT compilation of the shared cache directory.
        lock_path = os.path.join(build_dir, ".compile.lock")
        lock_fh = open(lock_path, "w", encoding="utf-8")
        try:
            try:
                import fcntl
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass

            # Re-check after acquiring lock in case another GPU process just compiled it
            if os.path.isfile(so_path):
                _CUDA_EXT = _load_so(so_path)
                _CUDA_ERROR = ""
                return _CUDA_EXT

            if shutil.which("ninja") is not None:
                from torch.utils.cpp_extension import load
                mod = load(
                    name=_EXT_NAME,
                    sources=[cu_path],
                    build_directory=build_dir,
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "-Xptxas=-O3"],
                    with_cuda=True,
                    verbose=False,
                    is_python_module=True,
                )
                _CUDA_EXT = mod
                _CUDA_ERROR = ""
                return _CUDA_EXT

            # Zero-dependency fallback when ninja is not installed: invoke nvcc directly
            _CUDA_EXT = _compile_direct_nvcc(cu_path, build_dir, so_path)
            _CUDA_ERROR = ""
            return _CUDA_EXT
        finally:
            try:
                import fcntl
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fh.close()
    except Exception as exc:
        _CUDA_EXT = None
        _CUDA_ERROR = str(exc)
        return None


def cuda_kernel_info() -> dict:
    """Return status metadata for the fused CUDA kernel extension."""
    return {
        "cuda_kernel_compiled": _CUDA_EXT is not None,
        "nvcc_detected": _find_nvcc() is not None,
        "cuda_kernel_error": _CUDA_ERROR,
    }


def is_low_fp64_gpu(dev: torch.device) -> bool:
    """Detect consumer/workstation GPUs with low FP64:FP32 ratio (e.g. L40S, L4, T4, RTX)."""
    if dev.type != "cuda":
        return False
    try:
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx).upper()
        for high_tag in ("A100", "H100", "H200", "B100", "B200", "V100", "GH200"):
            if high_tag in name:
                return False
        return True
    except Exception:
        return False
