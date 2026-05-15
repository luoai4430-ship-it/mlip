import sys
from pathlib import Path
import argparse
import time
import traceback
import subprocess
from functools import lru_cache
from pathlib import Path
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from collections import deque
from ase.io import read, write
from typing import Dict
import numpy as np
import dpdata

# 统一输出根目录：桌面文件夹
BASE_DIR = "C:/Users/Administrator/Desktop/MLIP_workspace"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR, exist_ok=True)

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="r Search MCP Server")
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

def _parse_wsl_path(work_dir: str) -> str:
    win_path = work_dir.replace('\\', '/')
    if "wsl.localhost/Ubuntu-24.04" in win_path:
        linux_path = win_path.split("wsl.localhost/Ubuntu-24.04")[1]
    elif "wsl$/Ubuntu-24.04" in win_path:
        linux_path = win_path.split("wsl$/Ubuntu-24.04")[1]
    elif win_path.startswith('C:') or win_path.startswith('c:'):
        linux_path = "/mnt/c" + win_path[2:]
    elif win_path.startswith('D:') or win_path.startswith('d:'):
        linux_path = "/mnt/d" + win_path[2:]
    else:
        linux_path = win_path
    if not linux_path.startswith("/"):
        linux_path = "/" + linux_path
    return linux_path

args = parse_args()
mcp = FastMCP("MLIP", port=args.port, host=args.host)

MODEL_CACHE = {}
def get_model(model_file, head=None):
    key = f"{model_file}_{head}"
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = DeepEval(str(model_file), head=head)
    return MODEL_CACHE[key]

@mcp.tool()
def health() -> str:
    """检查 Server 状态及根目录"""
    return json.dumps({
        "status": "ok", 
        "framework": "FastMCP Native",
        "base_directory": BASE_DIR
    }, ensure_ascii=False)

@mcp.tool()
def setup_project(project_name: str = "rare_earth") -> str:
    """在 MLIP_workspace下创建项目目录"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"{BASE_DIR}/{project_name}_{ts}"
    os.makedirs(path, exist_ok=True)
    return json.dumps({"project_path": path})



@mcp.tool()
def read_file_content(path: str, last_lines: int = 50) -> str:
    """
    高效且安全的读取文件工具。
    1. 默认读取最后 50 行。
    2. 使用 deque 防止大文件撑爆内存。
    3. 强制字符截断防止 LLM Token 溢出。
    """
    try:
        clean_path = os.path.normpath(path).replace('\\', '/')
        if not os.path.exists(clean_path):
            return json.dumps({"error": f"文件不存在: {clean_path}"}, ensure_ascii=False)
        
        with open(clean_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = deque(f, maxlen=last_lines)
            
        content = "".join(lines)
        
        MAX_CHARS = 8000 
        if len(content) > MAX_CHARS:
            content = "...[警告：内容过长，已被强制截断]...\n" + content[-MAX_CHARS:]
            
        return json.dumps({
            "status": "success",
            "file": clean_path,
            "content": content
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@mcp.tool()
def check_directory(path: str) -> str:
    """检查目录内容"""
    try:
        clean_path = os.path.normpath(path).replace('\\', '/')
        if not os.path.exists(clean_path): return json.dumps({"error": "路径不存在"})
        return json.dumps({"contents": os.listdir(clean_path)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

# ==========================================
# 🔥 核心科研工具：深度优化的解析与转换
# ==========================================

# ==========================================
# 🔥 Universal Scientific Dataset Backend
# ==========================================

SUPPORTED_FORMATS = {
    ".xyz": "extxyz",
    ".extxyz": "extxyz",
    ".cif": "cif",
    ".xml": "vasp/xml",
    ".dump": "lammps/dump",
    ".lammpstrj": "lammps/dump",
    ".npy": "deepmd/npy"
}


def detect_format(path: str) -> str:
    """
    自动检测数据格式
    """

    path = os.path.normpath(path)
    name = os.path.basename(path).upper()

    # =========================
    # VASP special files
    # =========================

    if name == "OUTCAR":
        return "vasp/outcar"
    if name in ["POSCAR", "CONTCAR"]:
        return "vasp/poscar"

    # =========================
    # DeepMD dataset directory
    # =========================

    if os.path.isdir(path):
        if os.path.exists(
            os.path.join(path, "type.raw")
        ):
            return "deepmd/npy"
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_FORMATS:
        return SUPPORTED_FORMATS[ext]

    raise ValueError(
        f"Unsupported format: {path}"
    )


# ==========================================
# Dataset Validation
# ==========================================

def validate_dataset(ds) -> dict:
    """
    数据集完整性检查
    """

    problems = []
    if isinstance(ds, dpdata.MultiSystems):
        if len(ds.systems) == 0:
            return {
                "valid": False,
                "problems": ["empty dataset"]
            }
        sample = list(ds.systems.values())[0]
        data = sample.data
    else:
        data = ds.data

    # =========================
    # Core checks
    # =========================

    required_fields = [
        "coords",
        "cells",
        "atom_names",
        "atom_types",
        "atom_numbs"
    ]

    for field in required_fields:
        if field not in data:
            problems.append(
                f"missing field: {field}"
            )

    # =========================
    # Scientific checks
    # =========================

    warnings = []
    if "energies" not in data:
        warnings.append("missing energies")
    if "forces" not in data:
        warnings.append("missing forces")

    # =========================
    # Shape consistency
    # =========================

    try:
        if "coords" in data:
            coords = data["coords"]
            if len(coords.shape) != 3:
                problems.append(
                    "coords shape invalid"
                )
    except Exception:
        problems.append(
            "coords corrupted"
        )

    return {
        "valid": len(problems) == 0,
        "problems": problems,
        "warnings": warnings
    }



# ==========================================
# Metadata Extraction
# ==========================================

def extract_metadata(ds) -> dict:
    """
    提取统一元数据
    """

    if isinstance(ds, dpdata.MultiSystems):
        sample = list(ds.systems.values())[0]
        data = sample.data
        nframes = sum(
            len(sys)
            for sys in ds.systems.values()
        )

    else:
        data = ds.data
        nframes = len(ds)

    return {
        "type_map":
            data.get("atom_names", []),
        "numb_atoms":
            sum(data.get("atom_numbs", [])),
        "numb_frames":
            nframes,
        "has_energy":
            "energies" in data,
        "has_force":
            "forces" in data,
        "has_virial":
            "virials" in data,
        "has_cell":
            "cells" in data
    }


# ==========================================
# ASE → dpdata bridge
# ==========================================

def ase_atoms_to_dpdata_system(atoms):
    """
    ASE Atoms → dpdata LabeledSystem
    """

    symbols = atoms.get_chemical_symbols()

    # 保持顺序稳定
    type_map = list(
        dict.fromkeys(symbols)
    )

    atom_types = np.array([
        type_map.index(s)
        for s in symbols
    ])

    atom_numbs = [
        symbols.count(sym)
        for sym in type_map
    ]

    data = {

        # =====================
        # REQUIRED
        # =====================

        "atom_names":
            type_map,
        "atom_numbs":
            atom_numbs,
        "atom_types":
            atom_types,
        "coords":
            np.array([
                atoms.positions
            ]),
        "cells":
            np.array([
                atoms.cell.array
            ]),
        # dpdata required
        "orig":
            np.array(
                [0.0, 0.0, 0.0]
            )
    }

    # =====================
    # Energy
    # =====================
    
# =====================
# Energy
# =====================

    energy_keys = [
        "energy",
        "Energy",
        "ENERGY",
        "free_energy",
        "FreeEnergy"
    ]

    for key in energy_keys:
        if key in atoms.info:
            data["energies"] = np.array([
                atoms.info[key]
            ])

            break

# =====================
# Forces
# =====================

    force_keys = [
        "forces",
        "force",
        "FORCES"
    ]

    for key in force_keys:
        if key in atoms.arrays:
            data["forces"] = np.array([
                atoms.arrays[key]
            ])

            break


# ==========================================
# Universal Dataset Loader
# ==========================================

def load_dataset(
    path: str,
    fmt: str = "auto"
):
    """
    统一科学数据加载入口
    """

    clean_path = os.path.normpath(path).replace(
        "\\",
        "/"
    )

    # =========================
    # Detect format
    # =========================

    if fmt == "auto":

        fmt = detect_format(
            clean_path
        )

    # =========================
    # DeepMD dataset
    # =========================

    if fmt == "deepmd/npy":
        ds = dpdata.MultiSystems.from_dir(
            dir_name=clean_path,
            file_name="*"
        )

        return ds, fmt

    # =========================
    # VASP labeled
    # =========================

    if fmt in [
        "vasp/outcar",
        "vasp/xml"
    ]:
        ds = dpdata.LabeledSystem(
            clean_path,
            fmt=fmt
        )

        return ds, fmt

    # =========================
    # POSCAR
    # =========================

    if fmt == "vasp/poscar":
        ds = dpdata.System(
            clean_path,
            fmt=fmt
        )
        return ds, fmt

    # =====================================
    # ASE-supported formats
    # =====================================

    atoms_list = read(
        clean_path,
        index=":"
    )

    # 单frame兼容
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    # =====================================
    # Check species consistency
    # =====================================

    ref_symbols = (
        atoms_list[0]
        .get_chemical_symbols()
    )
    same_species = all(
        at.get_chemical_symbols()
        == ref_symbols
        for at in atoms_list
    )

    # =====================================
    # Single System
    # =====================================

    if same_species:
        systems = [
            ase_atoms_to_dpdata_system(at)
            for at in atoms_list
        ]
        merged = systems[0]
        for sys in systems[1:]:
            merged.append(sys)

        return merged, fmt

    # =====================================
    # MultiSystems
    # =====================================

    ms = dpdata.MultiSystems()
    for at in atoms_list:
        ms.append(
            ase_atoms_to_dpdata_system(at)
        )

    return ms, fmt


# ==========================================
# MCP Tools
# ==========================================

@mcp.tool()
def parse_data(
    path: str,
    fmt: str = "auto"
) -> str:
    """
    通用科学数据解析器

    支持:
    - POSCAR
    - OUTCAR
    - vasprun.xml
    - xyz/extxyz
    - cif
    - lammps
    - deepmd/npy
    """

    try:
        ds, detected_fmt = load_dataset(
            path,
            fmt
        )
        validation = validate_dataset(ds)
        info = {
            "status": "success",
            "format_used":
                detected_fmt,
            "validation":
                validation,
            **extract_metadata(ds)
        }
        return json.dumps(
            info,
            ensure_ascii=False,
            indent=2
        )
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message":
                str(e)
        }, ensure_ascii=False)


@mcp.tool()
def convert_data(
    source_path: str,
    output_dir: str,
    target_fmt: str = "deepmd/npy"
) -> str:
    """
    通用科学数据转换器
    """

    try:
        ds, src_fmt = load_dataset(
            source_path
        )
        output_dir = os.path.normpath(
            output_dir
        ).replace("\\", "/")
        os.makedirs(
            output_dir,
            exist_ok=True
        )
        ds.to(
            target_fmt,
            output_dir
        )

        result = {
            "status": "success",
            "source_format":
                src_fmt,
            "target_format":
                target_fmt,
            "output_dir":
                output_dir,
            **extract_metadata(ds),
            "validation":
                validate_dataset(ds)
        }

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message":
                str(e)
        }, ensure_ascii=False)

if __name__ == "__main__":
    transport_type = os.getenv('MCP_TRANSPORT', 'streamable-http')
    mcp.run(transport=transport_type)
