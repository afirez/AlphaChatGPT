#!/usr/bin/env python3
import os
import sys
import re
import shutil
import subprocess
import time
from pathlib import Path

# ===================== 全局常量配置 (统一版本+联系方式，与原脚本完全一致) =====================
VERSION = "3.0"
CONTACT = "abus.aikorea@gmail.com"

# ===================== 【核心】分系统配置Miniconda信息 (原两个脚本的配置全部保留，精准对应) =====================
MINICONDA_CONFIG = {
    "win32": {
        "url": "https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Windows-x86_64.exe",
        "checksum": "978114c55284286957be2341ad0090eb5287222183e895bab437c4d1041a0284",
        "installer_name": "miniconda_installer.exe",
        "conda_bin": "_conda.exe",
        "python_bin": "python.exe"
    },
    "linux": {
        "url": "https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh",
        "checksum": "a95f99c31ee1d2bf87e51546b9c71f5820b792e05b0d2f4a1bc4618478efce15",
        "installer_name": "miniconda_installer.sh",
        "conda_bin": "conda",
        "python_bin": "python3"
    },
    "darwin": {
        "url": "https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-MacOSX-x86_64.sh",
        "checksum": "a95f99c31ee1d2bf87e51546b9c71f5820b792e05b0d2f4a1bc4618478efce15",
        "installer_name": "miniconda_installer.sh",
        "conda_bin": "conda",
        "python_bin": "python3"
    }
}

# 获取当前系统类型
CURRENT_OS = sys.platform
if CURRENT_OS not in MINICONDA_CONFIG:
    print(f"❌ 不支持的操作系统: {CURRENT_OS}")
    sys.exit(1)
OS_CONFIG = MINICONDA_CONFIG[CURRENT_OS]

# ===================== 公共工具函数 (所有系统共用，抽离核心逻辑，无冗余) =====================
def run_command(cmd, check=True, capture_output=False, silent=False):
    """跨平台命令执行函数，模拟原脚本的set -e/错误退出逻辑，兼容Windows/Linux/macOS"""
    if not silent:
        print(f"执行命令: {' '.join(cmd)}")
    shell_flag = True if CURRENT_OS == "win32" else False
    try:
        if capture_output:
            result = subprocess.run(
                cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                encoding="utf-8", shell=shell_flag
            )
        else:
            result = subprocess.run(cmd, check=check, shell=shell_flag)
        return result
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 命令执行失败: {' '.join(cmd)}")
        if capture_output:
            print(f"错误信息: {e.stderr.strip()}")
        if CURRENT_OS == "win32":
            input("\n按回车键退出...")
        sys.exit(1)

def print_big_message(msg_text):
    """公共函数：复刻原脚本的大警告信息打印，所有系统共用"""
    print("\n")
    print("*******************************************************************")
    print(f"* {msg_text}")
    print("*******************************************************************")
    print("\n")

def verify_file_checksum(file_path, expected_hash):
    """跨平台校验文件哈希值，自动适配Windows/Linux/macOS的校验工具"""
    if CURRENT_OS == "win32":
        # Windows: CertUtil -hashfile SHA256
        result = run_command(["CertUtil", "-hashfile", file_path, "SHA256"], capture_output=True, silent=True)
        hash_lines = [line.strip().upper() for line in result.stdout.split("\n") if line.strip()]
        actual_hash = ""
        for line in hash_lines:
            if re.match(r"^[0-9A-F]{64}$", line):
                actual_hash = line
                break
        return actual_hash == expected_hash.upper()
    else:
        # Linux/macOS: sha256sum
        result = run_command(["sha256sum", file_path], capture_output=True, silent=True)
        actual_hash = result.stdout.split()[0].strip().upper()
        return actual_hash == expected_hash.upper()

def find_conda_binary(conda_root):
    """跨平台查找conda二进制文件，适配Windows的_conda.exe和Linux/macOS的conda"""
    conda_root = Path(conda_root)
    conda_bin_name = OS_CONFIG["conda_bin"]
    
    # 标准路径1: conda_root/bin/conda  (Linux/macOS) / conda_root/_conda.exe (Windows)
    conda_bin1 = conda_root / conda_bin_name if CURRENT_OS == "win32" else conda_root / "bin" / conda_bin_name
    if conda_bin1.exists() and os.access(conda_bin1, os.X_OK):
        return str(conda_bin1)
    
    # 标准路径2: conda_root/condabin/conda (Linux/macOS)
    if CURRENT_OS != "win32":
        conda_bin2 = conda_root / "condabin" / conda_bin_name
        if conda_bin2.exists() and os.access(conda_bin2, os.X_OK):
            return str(conda_bin2)
    
    # pkgs目录兜底查找
    pkgs_conda = ""
    glob_pattern = "pkgs/**/" + conda_bin_name if CURRENT_OS == "win32" else "pkgs/**/bin/conda"
    for p in conda_root.glob(glob_pattern):
        if p.is_file() and os.access(p, os.X_OK):
            pkgs_conda = str(p)
            break
    if pkgs_conda:
        return pkgs_conda
    
    # 全局兜底查找
    found_conda = ""
    glob_pattern = conda_bin_name if CURRENT_OS == "win32" else "**/bin/conda"
    for p in conda_root.rglob(glob_pattern):
        if p.is_file() and os.access(p, os.X_OK):
            found_conda = str(p)
            break
    return found_conda if found_conda else ""

def check_conda_base_python(conda_root):
    """跨平台校验conda的base python是否损坏，所有系统逻辑一致"""
    conda_root = Path(conda_root)
    python_bin_name = OS_CONFIG["python_bin"]
    python_bin = conda_root / python_bin_name if CURRENT_OS == "win32" else conda_root / "bin" / python_bin_name
    if not python_bin.exists():
        return False
    
    test_cmds = ["import sys; import os; import math", "print('OK')"]
    for cmd in test_cmds:
        try:
            result = subprocess.run([str(python_bin), "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
            if result.returncode != 0 or (cmd == test_cmds[-1] and "OK" not in result.stdout):
                return False
        except:
            return False
    return True

# ===================== Windows特有函数 (仅Windows执行，Linux/macOS跳过) =====================
def windows_64bit_env_check():
    """Windows专属：64位环境强制校验+自动重启修复，原bat脚本的核心逻辑"""
    if CURRENT_OS != "win32":
        return
    is_64bit_os = sys.maxsize > 2**32
    processor_arch = os.environ.get("PROCESSOR_ARCHITECTURE", "").upper()
    processor_arch_w6432 = os.environ.get("PROCESSOR_ARCHITEW6432", "").upper()
    
    # 32位CMD运行在64位系统 → 重启到Sysnative 64位CMD
    if processor_arch == "X86" and processor_arch_w6432:
        sysnative_cmd = os.path.join(os.environ["SystemRoot"], "Sysnative", "cmd.exe")
        script_path = os.path.abspath(__file__)
        print(f"检测到32位CMD在64位系统运行，自动重启到64位环境...")
        subprocess.run([sysnative_cmd, "/c", script_path] + sys.argv[1:], shell=False)
        sys.exit(0)
    
    # 从SysWOW64目录运行 → 重启到System32 CMD
    script_dir = os.path.dirname(os.path.abspath(__file__)).upper()
    syswow64_dir = os.path.join(os.environ["SystemRoot"], "SysWOW64").upper()
    if script_dir == syswow64_dir:
        system32_cmd = os.path.join(os.environ["SystemRoot"], "System32", "cmd.exe")
        script_path = os.path.abspath(__file__)
        print(f"检测到从SysWOW64运行，自动切换到System32环境...")
        subprocess.run([system32_cmd, "/c", script_path] + sys.argv[1:], shell=False)
        sys.exit(0)
    print("✅ Windows 64位运行环境校验通过")

# ===================== 主程序入口 (所有系统共用，分支执行系统特有逻辑) =====================
def main():
    # ===================== 1. 系统特有前置校验 =====================
    if CURRENT_OS == "win32":
        windows_64bit_env_check()
    else:
        print(f"✅ 检测到系统: {CURRENT_OS.upper()} (Linux/macOS)")

    # ===================== 2. 打印头部信息 (完全复刻原脚本) =====================
    print("=========================================================================")
    print("")
    print(f"  ABUS Launcher [Version {VERSION}]")
    print(f"  contact: {CONTACT}")
    print("")
    print("=========================================================================")
    print("")

    # ===================== 3. 切换到脚本目录 + 路径检测 (公共逻辑) =====================
    script_path = Path(os.path.abspath(__file__)).resolve()
    script_dir = str(script_path.parent)
    os.chdir(script_dir)
    print(f"运行目录: {script_dir}")
    print(f"命令行参数: {' '.join(sys.argv[1:])}")
    print("")

    # 路径含空格 → 退出
    if " " in script_dir:
        print("This script relies on Miniconda which can not be silently installed under a path with spaces.")
        if CURRENT_OS == "win32":
            input("\n按回车键退出...")
        sys.exit(1)

    # 路径含特殊字符 → 警告
    special_char_pattern = r'[!#\$%&()\*+,;<=>?@\[\]\^`{|}~]'
    if re.search(special_char_pattern, script_dir):
        warn_msg = "WARNING: Special characters were detected in the installation path!         This can cause the installation to fail!"
        print_big_message(warn_msg)

    # ===================== 4. 公共路径配置 + 临时目录 =====================
    INSTALL_DIR = os.path.join(script_dir, "installer_files")
    CONDA_ROOT_PREFIX = os.path.join(INSTALL_DIR, "conda")
    INSTALL_ENV_DIR = os.path.join(INSTALL_DIR, "env")
    os.makedirs(INSTALL_DIR, exist_ok=True)
    os.environ["TMP"] = INSTALL_DIR
    os.environ["TEMP"] = INSTALL_DIR

    # ===================== 5. 清理conda环境冲突 (公共逻辑) =====================
    print("清理现有conda环境，避免冲突...")
    for _ in range(3):
        run_command(["conda", "deactivate"], check=False, silent=True)
    print("")

    # ===================== 6. 检测conda是否已安装 (公共逻辑) =====================
    CONDA_BIN = find_conda_binary(CONDA_ROOT_PREFIX)
    CONDA_EXISTS = False
    CONDA_BASE_CORRUPTED = False

    if CONDA_BIN and os.path.isfile(CONDA_BIN):
        CONDA_EXISTS = True
        print("Checking conda base Python installation...")
        if not check_conda_base_python(CONDA_ROOT_PREFIX):
            print("WARNING: Conda base Python installation appears corrupted.")
            print("This can happen if the installation was interrupted or files were corrupted.")
            python_bin_name = OS_CONFIG["python_bin"]
            python_bin = os.path.join(CONDA_ROOT_PREFIX, python_bin_name) if CURRENT_OS == "win32" else os.path.join(CONDA_ROOT_PREFIX, "bin", python_bin_name)
            print(f"  - Python binary: {python_bin}")
            print(f"  - Python exists: {'YES' if os.path.exists(python_bin) else 'NO'}")
            CONDA_BASE_CORRUPTED = True
        else:
            print("Conda base Python installation verified.")

    # ===================== 7. 跨平台安装Miniconda (分支执行系统特有命令) =====================
    installer_path = os.path.join(INSTALL_DIR, OS_CONFIG["installer_name"])
    if not CONDA_EXISTS or CONDA_BASE_CORRUPTED:
        if CONDA_BASE_CORRUPTED:
            print("Removing corrupted conda installation...")
            if os.path.exists(CONDA_ROOT_PREFIX):
                shutil.rmtree(CONDA_ROOT_PREFIX)
            print("Conda installation removed. Will reinstall...")
        
        print(f"开始下载Miniconda: {OS_CONFIG['url']}")
        run_command(["curl", "-Lk", OS_CONFIG["url"], "-o", installer_path], silent=True)
        if not os.path.exists(installer_path):
            print("❌ Miniconda下载失败！")
            if CURRENT_OS == "win32":
                input("\n按回车键退出...")
            sys.exit(1)
        
        # 校验哈希值
        print("开始校验安装包哈希值...")
        if not verify_file_checksum(installer_path, OS_CONFIG["checksum"]):
            print(f"❌ {OS_CONFIG['installer_name']} 哈希值校验失败！")
            os.remove(installer_path)
            if CURRENT_OS == "win32":
                input("\n按回车键退出...")
            sys.exit(1)
        else:
            print(f"✅ {OS_CONFIG['installer_name']} 哈希值校验成功！")
        
        # 分系统执行静默安装
        print(f"开始静默安装Miniconda到: {CONDA_ROOT_PREFIX}")
        if CURRENT_OS == "win32":
            # Windows exe静默安装参数 (原bat脚本参数完全一致)
            install_args = [
                installer_path, "/InstallationType=JustMe", "/NoShortcuts=1", "/AddToPath=0",
                "/RegisterPython=0", "/NoRegistry=1", "/S", f"/D={CONDA_ROOT_PREFIX}"
            ]
            run_command(install_args, silent=True)
        else:
            # Linux/macOS sh脚本静默安装参数 (原sh脚本参数完全一致)
            run_command(["bash", installer_path, "-b", "-p", CONDA_ROOT_PREFIX, "-u"], silent=True)
        
        # 校验安装结果
        print("Miniconda 版本信息:")
        CONDA_BIN = find_conda_binary(CONDA_ROOT_PREFIX)
        run_command([CONDA_BIN, "--version"], silent=False)
        
        # 删除安装包
        if os.path.exists(installer_path):
            os.remove(installer_path)
            print("✅ 已删除安装包，释放空间")
        print("")

    # ===================== 8. 创建conda环境 (Python3.10) 公共逻辑 =====================
    abus_genuine_installed = True
    if not os.path.exists(INSTALL_ENV_DIR):
        abus_genuine_installed = False
        print("开始创建conda环境 (Python3.10)...")
        run_command([CONDA_BIN, "create", "--no-shortcuts" if CURRENT_OS == "win32" else "", "-y", "-k", "--prefix", INSTALL_ENV_DIR, "python=3.10"], silent=True)
        print("✅ conda环境创建完成")
    else:
        print("✅ 检测到已存在conda环境")
    print("")

    # ===================== 9. 校验环境有效性 (公共逻辑) =====================
    python_bin_name = OS_CONFIG["python_bin"]
    PYTHON_EXE_PATH = os.path.join(INSTALL_ENV_DIR, python_bin_name) if CURRENT_OS == "win32" else os.path.join(INSTALL_ENV_DIR, "bin", python_bin_name)
    if not os.path.exists(PYTHON_EXE_PATH):
        print("❌ Conda environment is empty. Python可执行文件不存在！")
        if CURRENT_OS == "win32":
            input("\n按回车键退出...")
        sys.exit(1)

    # ===================== 10. 环境隔离配置 (公共逻辑，完全一致) =====================
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONPATH"] = ""
    os.environ["PYTHONHOME"] = ""
    os.environ["CUDA_PATH"] = INSTALL_ENV_DIR
    os.environ["CUDA_HOME"] = INSTALL_ENV_DIR
    print("✅ 环境隔离配置完成")
    print("")

    # ===================== 11. 激活环境 + 安装依赖 + 启动服务 (跨平台适配) =====================
    print(f"Miniconda 安装位置: {CONDA_ROOT_PREFIX}")
    os.chdir(script_dir)

    # 激活脚本路径适配
    if CURRENT_OS == "win32":
        conda_activate_script = os.path.join(CONDA_ROOT_PREFIX, "condabin", "conda.bat")
        activate_cmd = f'"{conda_activate_script}" activate "{INSTALL_ENV_DIR}" && '
    else:
        conda_activate_script = os.path.join(CONDA_ROOT_PREFIX, "etc", "profile.d", "conda.sh")
        activate_cmd = f"source {conda_activate_script} && conda activate {INSTALL_ENV_DIR} && "

    # 安装依赖
    if not abus_genuine_installed:
        print("开始安装依赖: huggingface-hub==0.27.1")
        run_command(f'{activate_cmd} python -m pip install huggingface-hub==0.27.1', silent=True)
        print("✅ 依赖安装完成")
        print("")

    # 启动核心服务
    os.environ["LOG_LEVEL"] = "DEBUG"
    print("=== 开始启动 ABUS Voice Service ===")
    print("====================================")
    run_command(f'{activate_cmd} python start-abus.py voice', silent=False)

    # 脚本结束
    print("\n✅ ABUS服务执行完成！")
    if CURRENT_OS == "win32":
        input("\n按回车键退出...")

if __name__ == "__main__":
    if sys.version_info < (3, 0):
        print("错误: 本脚本需要Python 3.x 版本运行！")
        if CURRENT_OS == "win32":
            input("\n按回车键退出...")
        sys.exit(1)
    main()