# front-streaminng/gpu_stats.py
from __future__ import annotations
import shutil
import subprocess
from typing import Optional, Tuple

# ---- NVIDIA NVML を使えるなら最優先で使う（高速・正確） ----
_NVML_AVAILABLE = False
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetName, nvmlDeviceGetCount
    )
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


def init_nvml_once() -> bool:
    """NVMLが利用可能なら初期化して True を返す。非対応/失敗時は False。"""
    global _NVML_AVAILABLE
    if not _NVML_AVAILABLE:
        return False
    try:
        # すでに初期化済みなら例外にならない（pynvmlは内部で参照カウント持つ）。
        nvmlInit()
        return True
    except Exception:
        return False


def shutdown_nvml() -> None:
    """可能ならNVMLを終了（他プロセス/サーバを停止するわけではありません）。"""
    if not _NVML_AVAILABLE:
        return
    try:
        nvmlShutdown()
    except Exception:
        pass


# ---------- 実装詳細：各ベンダ/手段ごとの取得 ----------
def _nvml_get() -> Optional[Tuple[float, float, int, str]]:
    """NVMLで (used_GB, total_GB, util_percent, gpu_name) を返す。"""
    if not _NVML_AVAILABLE:
        return None
    try:
        count = nvmlDeviceGetCount()
        if count == 0:
            return None
        h = nvmlDeviceGetHandleByIndex(0)  # 複数GPUならループで集計も可
        mem = nvmlDeviceGetMemoryInfo(h)
        util = nvmlDeviceGetUtilizationRates(h)
        name = nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else str(name)
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        gpu_util = int(util.gpu)
        return used_gb, total_gb, gpu_util, name
    except Exception:
        return None


def _nvsmi_get() -> Optional[Tuple[float, float, int, str]]:
    """nvidia-smi（CLI）で (used_GB, total_GB, util_percent, gpu_name) を返す。"""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]
        used_mb, total_mb, util_p, name = [x.strip() for x in out.split(",")]
        return float(used_mb) / 1024.0, float(total_mb) / 1024.0, int(util_p), name
    except Exception:
        return None


def _rocm_get() -> Optional[Tuple[float, float, int, str]]:
    """rocm-smi（CLI）で (used_GB, total_GB, util_percent, gpu_name) を返す。AMD向け。"""
    if not shutil.which("rocm-smi"):
        return None
    try:
        import json
        mem_out = subprocess.check_output(["rocm-smi", "--showmemuse", "--json"], text=True)
        util_out = subprocess.check_output(["rocm-smi", "--showuse", "--json"], text=True)
        mem = json.loads(mem_out)
        util = json.loads(util_out)
        gpu = next(iter(mem))  # 単一GPU想定（必要に応じて拡張）
        used_mib = mem[gpu]["VRAM Used (MiB)"]
        total_mib = mem[gpu]["VRAM Total (MiB)"]
        util_p = util[gpu]["GPU use (%)"]
        name = gpu
        return float(used_mib) / 1024.0, float(total_mib) / 1024.0, int(util_p), name
    except Exception:
        return None


# ---------- 公開関数 ----------
def get_gpu_stats() -> Optional[Tuple[float, float, int, str]]:
    """
    利用可能な方法で VRAM を取得して返す。
    返り値: (used_GB, total_GB, util_percent, gpu_name)
    取得不可なら None。
    """
    return _nvml_get() or _nvsmi_get() or _rocm_get()