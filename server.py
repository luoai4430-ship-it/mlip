import argparse
import time
import json
import os
from mcp.server.fastmcp import FastMCP
from collections import deque

# 统一输出根目录：桌面文件夹
BASE_DIR = "C:/Users/Administrator/Desktop/MLIP_workspace"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR, exist_ok=True)

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="r MCP Server")
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
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

def _lazy_import_science_libs():
    """Lazy import heavy scientific libraries to reduce MCP startup cost."""
    import numpy as np
    import dpdata
    from ase.io import read
    return np, dpdata, read

def get_model(model_file, head=None):
    from deepmd.infer import DeepEval
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
    _, dpdata, _ = _lazy_import_science_libs()
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

    _, dpdata, _ = _lazy_import_science_libs()
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

    np, dpdata, _ = _lazy_import_science_libs()
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
    energy = None
    # Case 1:
    # atoms.info
    for key in [
        "energy",
        "Energy",
        "ENERGY",
        "free_energy",
        "FreeEnergy"
    ]:
        if key in atoms.info:

            energy = atoms.info[key]
            break

    # Case 2:
    # ASE calculator results
    if energy is None:
        try:
            energy = atoms.get_potential_energy()
        except Exception:

            pass

    # Save
    if energy is not None:
        data["energies"] = np.array([
            energy
        ])

    # =====================
    # Forces
    # =====================
    forces = None
    # Case 1:
    # arrays
    for key in [
        "forces",
        "force",
        "FORCES"
    ]:

        if key in atoms.arrays:

            forces = atoms.arrays[key]
            break
    # Case 2:
    # calculator
    if forces is None:
        try:
            forces = atoms.get_forces()
        except Exception:
            pass

    # Save
    if forces is not None:
        data["forces"] = np.array([
            forces
        ])

    # 有能量/力时返回 LabeledSystem，否则返回 System
    if "energies" not in data:
        data["energies"] = np.array([
            0.0
        ])
    if "forces" not in data:
        natoms = len(symbols)
        data["forces"] = np.zeros(
            (1, natoms, 3)
        )

    return dpdata.LabeledSystem(
        data=data
)

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

    _, dpdata, read = _lazy_import_science_libs()
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


@mcp.tool()
def create_input_json(
        project_dir: str,
        type_map: list,
        training_systems: list,
        validation_systems: list,
        descriptor: dict = None,
        fitting_net: dict = None,
        learning_rate: dict = None,
        loss: dict = None,
        test_mode: bool = True
) -> str:
    """
    通用 DeepMD input.json 生成器
    """

    try:
        cwd = os.path.normpath(project_dir).replace("\\", "/")
        os.makedirs(cwd, exist_ok=True)

        # 默认 descriptor
        if descriptor is None:
            descriptor = {
                "type": "se_e2_a",
                "sel": [60],
                "rcut_smth": 0.5,
                "rcut": 6.0,
                "neuron": [25, 50, 100],
                "resnet_dt": False,
                "axis_neuron": 16,
                "seed": 1
            }

        # 默认 fitting_net
        if fitting_net is None:
            fitting_net = {
                "neuron": [120, 120, 120],
                "resnet_dt": True,
                "seed": 1
            }

        # 默认 learning rate
        if learning_rate is None:
            learning_rate = {
                "type": "exp",
                "decay_steps": 5000,
                "start_lr": 0.001,
                "stop_lr": 3.51e-8
            }

        # 默认 loss
        if loss is None:
            loss = {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0
            }

        # 训练模式
        numb_steps = 1000 if test_mode else 1000000
        disp_freq = 10 if test_mode else 1000
        save_freq = 50 if test_mode else 5000

        config = {
            "model": {
                "type_map": type_map,
                "descriptor": descriptor,
                "fitting_net": fitting_net
            },
            "learning_rate": learning_rate,
            "loss": loss,
            "training": {
                "training_data": {
                    "systems": training_systems,
                    "batch_size": "auto"
                },
                "validation_data": {
                    "systems": validation_systems,
                    "batch_size": 1,
                    "numb_btch": 1
                },
                "numb_steps": numb_steps,
                "seed": 1,
                "disp_file": "lcurve.out",
                "disp_freq": disp_freq,
                "save_freq": save_freq,
                "tensorboard": True,
                "tensorboard_log_dir": "log",
                "tensorboard_freq": disp_freq
            }
        }

        file_path = os.path.join(cwd, "input.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        return json.dumps({
            "status": "success",
            "config_file": file_path,
            "type_map": type_map,
            "descriptor": descriptor,
            "fitting_net": fitting_net
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, ensure_ascii=False)


@mcp.tool()
def update_deepmd_config(file_path: str, updates_json: str) -> str:
    """
    JSON 配置文件修改工具，专为模型微调设计。
    支持批量、深度嵌套修改，支持直接写入列表（数组）和字典。

    参数:
    - file_path: input.json 的绝对或相对路径
    - updates_json: 包含所有修改项的 JSON 字符串，键名使用点号(.)表示层级。
      例如: '{"loss.type": "ener", "training.numb_steps": 1000, "model.descriptor.neuron": [25, 50, 100]}'
    """

    try:
        clean_path = os.path.normpath(file_path).replace('\\', '/')
        if not os.path.exists(clean_path):
            return json.dumps({"error": f"找不到文件: {clean_path}"}, ensure_ascii=False)

        # 1. 加载原配置
        with open(clean_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 解析大模型传过来的批量修改指令
        updates = json.loads(updates_json)

        # 3. 遍历并执行深度修改
        for key_path, new_value in updates.items():
            keys = key_path.split('.')
            current = data

            # 深入到倒数第一层
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}  # 如果路径不存在，自动创建字典
                current = current[key]

            # 在最后一层赋值（无论是数字、字符串、列表还是新字典，都能完美替换）
            current[keys[-1]] = new_value

        # 4. 写回文件
        with open(clean_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        return json.dumps({
            "status": "success",
            "message": "配置已成功更新",
            "updated_keys": list(updates.keys())
        }, ensure_ascii=False)

    except json.JSONDecodeError:
        return json.dumps({"error": "updates_json 格式错误，必须是合法的 JSON 字符串"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"修改失败: {str(e)}"}, ensure_ascii=False)


# ==========================================
# 🚀 闭环训练与评估
# ==========================================

@mcp.tool()
def train_and_freeze_async(project_dir: str) -> str:
    """
    在后台启动 DeepMD 训练和冻结任务。
    """
    import subprocess
    try:
        cwd = os.path.normpath(project_dir).replace('\\', '/')
        log_path = f"{cwd}/mcp_train.log"

        # 记录训练的原始日志
        log_file = open(log_path, "w")
        cmd_train = "dp train input.json && dp freeze -o graph.pb"

        # 启动后台训练
        subprocess.Popen(
            cmd_train, shell=True, cwd=cwd, stdout=log_file, stderr=log_file,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

        msg = f"🚀 训练已在后台纯净启动！日志文件: {log_path}\n💡 建议等待几分钟。"

        return json.dumps({"status": "started", "message": msg}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"训练启动失败: {str(e)}"}, ensure_ascii=False)


@mcp.tool()
def evaluate_model(data_dir: str, model_file: str) -> str:
    """
    计算验证集的能量、单原子能量及力 RMSE。
    这是模型微调闭环中的核心“质检”工具。
    """
    import numpy as np
    import dpdata
    try:
        clean_data_dir = os.path.normpath(data_dir).replace('\\', '/')
        clean_model_file = os.path.normpath(model_file).replace('\\', '/')

        if not os.path.exists(clean_data_dir):
            return json.dumps({"error": f"找不到验证集数据: {clean_data_dir}"}, ensure_ascii=False)
        if not os.path.exists(clean_model_file):
            return json.dumps({"error": f"找不到冻结的模型文件: {clean_model_file}"}, ensure_ascii=False)

        data = dpdata.MultiSystems()
        data.load_systems_from_file(clean_data_dir, fmt="deepmd/npy")
        model = get_model(clean_model_file)

        e_sq_err, e_pa_sq_err, f_sq_err = [], [], []
        total_frames = 0

        for sys in data:
            natoms = len(sys.data["atom_types"])
            total_frames += len(sys)

            # 推理：同时获取预测的能量、力和维里
            e_pred, f_pred, v_pred = model.eval(
                sys.data["coords"].reshape([len(sys), -1, 3]),
                cells=sys.data.get("cells"),
                atom_types=sys.data["atom_types"].reshape([1, -1])
            )

            # 1. 能量误差收集
            e_gt = sys.data["energies"].flatten()
            e_diff = e_pred.flatten() - e_gt
            e_sq_err.extend(e_diff ** 2)
            e_pa_sq_err.extend((e_diff / natoms) ** 2)  # 单原子能量平方误差

            # 2. 力误差收集 (如果验证集中包含真实力数据)
            if "forces" in sys.data and f_pred is not None:
                f_gt = sys.data["forces"].flatten()
                f_diff = f_pred.flatten() - f_gt
                f_sq_err.extend(f_diff ** 2)

        # 计算最终 RMSE
        rmse_e = float(np.sqrt(np.mean(e_sq_err)))
        rmse_e_per_atom = float(np.sqrt(np.mean(e_pa_sq_err)))
        rmse_f = float(np.sqrt(np.mean(f_sq_err))) if f_sq_err else None

        return json.dumps({
            "status": "success",
            "metrics": {
                "rmse_energy_total": round(rmse_e, 5),
                "rmse_energy_per_atom": round(rmse_e_per_atom, 6),
                "rmse_force": round(rmse_f, 5) if rmse_f else "N/A"
            },
            "info": {
                "tested_frames": total_frames,
                "model": os.path.basename(clean_model_file)
            }
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"评估失败: {str(e)}"}, ensure_ascii=False)


@mcp.tool()
def plot_lammps_thermo(work_dir: str, log_file: str = "log.lammps", output_image: str = "thermo_plot.png") -> str:
    """
    读取项目文件夹下的 LAMMPS 日志文件，并绘制热力学收敛图。
    """
    import matplotlib.pyplot as plt

    try:
        # 1. 直接使用原生项目文件夹路径拼接
        log_path = os.path.join(work_dir, log_file)
        img_path = os.path.join(work_dir, output_image)

        if not os.path.exists(log_path):
            return json.dumps({"error": f"找不到日志文件: {log_path}，请确认 MD 模拟是否成功生成了日志。"},
                              ensure_ascii=False)

        # 2. 智能提取 LAMMPS 热力学数据
        steps, temps, pes, kes, etotals = [], [], [], [], []
        is_reading_thermo = False

        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                # 寻找表头
                if line.startswith("Step Temp Pe Ke Etotal"):
                    is_reading_thermo = True
                    continue
                # 遇到循环结束的标志，停止读取
                if line.startswith("Loop time of"):
                    is_reading_thermo = False
                    continue

                # 提取纯数字数据行
                if is_reading_thermo:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            steps.append(float(parts[0]))
                            temps.append(float(parts[1]))
                            pes.append(float(parts[2]))
                            kes.append(float(parts[3]))
                            etotals.append(float(parts[4]))
                        except ValueError:
                            pass  # 忽略非数字的干扰行

        if not steps:
            return json.dumps({"error": f"未在 {log_file} 中提取到有效的热力学数据！"}, ensure_ascii=False)

        # 3. 绘制出版级的高清双子图 (温度图 + 能量图)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 上半部分：温度波动图
        ax1.plot(steps, temps, color='#FF6B6B', linewidth=2, label='Temperature (K)')
        ax1.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
        ax1.set_title('Molecular Dynamics Thermodynamic Properties', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right')

        # 下半部分：能量守恒图
        ax2.plot(steps, pes, color='#4ECDC4', linewidth=2, label='Potential Energy (eV)')
        ax2.plot(steps, kes, color='#FFE66D', linewidth=2, label='Kinetic Energy (eV)')
        ax2.plot(steps, etotals, color='#292F36', linewidth=2, linestyle='--', label='Total Energy (eV)')
        ax2.set_xlabel('Simulation Steps', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Energy (eV)', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='center right')

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 返回结果给 Agent
        return json.dumps({
            "status": "success",
            "message": f"📊 热力学可视化已完成！\n提取了 {len(steps)} 步的数据。\n高清图表已成功保存至项目文件夹: {img_path}"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"可视化工具执行崩溃: {str(e)}"}, ensure_ascii=False)


@mcp.tool()
def run_docker_lammps_md(work_dir: str, model_file: str = "graph.pb", temp: float = 300.0, steps: int = 100) -> str:
    """
    使用 Docker + GPU 加速在后台运行分子动力学 (MD) 模拟。
    该工具提交任务后会立即返回，不会导致超时。
    """
    import subprocess
    try:
        # 1. 路径极速翻译
        linux_path = _parse_wsl_path(work_dir)

        # 2. 生成 LAMMPS 运行脚本 (注：官方镜像通常不需要显式 plugin load，此处保留但建议核实)
        lammps_script = f"""
clear
units metal
boundary p p p
atom_style atomic

plugin load libdeepmd_lmp.so

lattice fcc 5.30
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 138.905

pair_style deepmd {model_file}
pair_coeff * *

velocity all create {temp} 87287 loop geom
fix 1 all nvt temp {temp} {temp} 0.1

thermo 10
thermo_style custom step temp pe ke etotal press
thermo_modify format float %15.8g

# 输出动画轨迹文件
dump 1 all custom 50 la_md.lammpstrj id type x y z

run {steps}
"""
        # 写入 IN 文件
        md_in_path = os.path.join(work_dir, "in.md")
        with open(md_in_path, "w") as f:
            f.write(lammps_script)

        # 定义日志文件路径
        log_file_name = "lammps_run.log"
        log_file_win = os.path.join(work_dir, log_file_name).replace('\\', '/')

        # 3. 构造后台运行指令
        # 使用 > {log_file_name} 2>&1 将所有输出重定向到文件
        docker_image = "deepmodeling/deepmd-kit:3.1.0_cuda129"

        cmd = (
            f'wsl -d Ubuntu-24.04 bash -c "'
            f'cd \'{linux_path}\' && '
            f'docker run --gpus all --rm -v \\$(pwd):/work -w /work {docker_image} lmp -in in.md > {log_file_name} 2>&1"'
        )

        # 4. 使用 Popen 异步启动
        # stdout/stderr 设置为 DEVNULL，因为输出已经重定向到 log 文件中了
        subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )

        # 5. 立即返回状态
        return json.dumps({
            "status": "started",
            "message": "✅ LAMMPS 模拟任务已成功提交至后台运行，避开了超时限制。",
            "work_dir": work_dir,
            "log_file": log_file_win,
            "instruction": f"模拟正在进行中。请在几分钟后使用读取文件工具检查 {log_file_win}。当日志末尾出现 'Total wall time' 时表示计算完成。"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"MCP 工具提交失败: {str(e)}"}, ensure_ascii=False)

if __name__ == "__main__":
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)
