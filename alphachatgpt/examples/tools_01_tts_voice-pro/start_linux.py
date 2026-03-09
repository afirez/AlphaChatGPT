#!/usr/bin/env python3
import os
import sys
import re
import shutil
import subprocess
import platform
import time
from pathlib import Path

# 全局配置，和原shell脚本完全一致
VERSION = "3.0"
CONTACT = "abus.aikorea@gmail.com"

def run_command(cmd, check=True, capture_output=False, silent=False):
    """
    执行系统命令，模拟shell的set -e，命令失败立即退出脚本
    :param cmd: 命令列表(list)
    :param check: True=失败则退出，False=忽略失败
    :param capture_output: True=捕获stdout/stderr，返回结果对象；False=直接输出到控制台
    :param silent: True=不打印命令内容，False=打印执行的命令
    :return: subprocess.CompletedProcess 对象
    """
    if not silent:
        print(f"Exec: {' '.join(cmd)}")
    try:
        if capture_output:
            result = subprocess.run(
                cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
            )
        else:
            result = subprocess.run(cmd, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        if capture_output:
            print(f"Error Info: {e.stderr.strip()}")
        sys.exit(1)

def find_conda_binary(conda_root):
    """
    复刻原shell的find_conda_binary函数，优先级完全一致
    查找顺序: conda_root/bin/conda > conda_root/condabin/conda > pkgs下的conda > 全局查找
    """
    conda_root = Path(conda_root)
    # 标准路径1
    conda_bin1 = conda_root / "bin" / "conda"
    if conda_bin1.exists() and os.access(conda_bin1, os.X_OK):
        return str(conda_bin1)
    
    # 标准路径2
    conda_bin2 = conda_root / "condabin" / "conda"
    if conda_bin2.exists() and os.access(conda_bin2, os.X_OK):
        return str(conda_bin2)
    
    # pkgs目录查找
    pkgs_conda = ""
    for p in conda_root.glob("pkgs/**/bin/conda"):
        if p.is_file() and os.access(p, os.X_OK):
            pkgs_conda = str(p)
            break
    if pkgs_conda:
        return pkgs_conda
    
    # 最后兜底：全局查找conda_root下的bin/conda
    found_conda = ""
    for p in conda_root.rglob("bin/conda"):
        if p.is_file() and os.access(p, os.X_OK):
            found_conda = str(p)
            break
    return found_conda if found_conda else ""

def check_conda_base_python(conda_root):
    """
    复刻原shell的check_conda_base_python函数，校验conda的base python是否损坏
    返回: True=正常, False=损坏
    """
    conda_root = Path(conda_root)
    python_bin = conda_root / "bin" / "python"
    if not python_bin.exists():
        return False
    
    # 测试内置模块导入 + 基础执行
    test_cmds = [
        "import sys; import os; import math",
        "print('OK')"
    ]
    for cmd in test_cmds:
        try:
            result = subprocess.run(
                [str(python_bin), "-c", cmd],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
            )
            if result.returncode != 0:
                return False
            if cmd == test_cmds[-1] and "OK" not in result.stdout:
                return False
        except:
            return False
    return True

def main():
    # ===== 打印头部信息，和原脚本完全一致 =====
    print("=========================================================================")
    print("")
    print(f"  ABUS Launcher [Version {VERSION}]")
    print(f"  contact: {CONTACT}")
    print("")
    print("=========================================================================")
    print("")

    # ===== 获取脚本目录并切换，复刻原SCRIPT_DIR逻辑 =====
    script_path = Path(os.path.abspath(__file__)).resolve()
    script_dir = script_path.parent
    os.chdir(script_dir)
    SCRIPT_DIR = str(script_dir)

    # ===== 检测路径是否包含空格，有则退出，和原脚本一致 =====
    if " " in SCRIPT_DIR:
        print("This script relies on Miniconda which can not be silently installed under a path with spaces.")
        sys.exit(1)

    # ===== 检测路径是否包含特殊字符，警告提示，和原脚本一致 =====
    special_chars = r'[!@#\$%^\&*\(\)+,\;:=\<\>\?@\[\]\^\`\{\|\}~]'
    if re.search(special_chars, SCRIPT_DIR):
        print("")
        print("*******************************************************************")
        print("* WARNING: Special characters were detected in the installation path!")
        print("*          This can cause the installation to fail!")
        print("*******************************************************************")
        print("")

    # ===== 设置路径常量，和原脚本完全一致 =====
    INSTALL_DIR = os.path.join(SCRIPT_DIR, "installer_files")
    CONDA_ROOT_PREFIX = os.path.join(INSTALL_DIR, "conda")
    INSTALL_ENV_DIR = os.path.join(INSTALL_DIR, "env")

    # ===== 设置临时目录环境变量 =====
    os.environ["TMP"] = INSTALL_DIR
    os.environ["TEMP"] = INSTALL_DIR

    # ===== 检测操作系统，设置Miniconda下载信息 =====
    os_type = platform.system()
    MINICONDA_URL = ""
    MINICONDA_CHECKSUM = ""
    MINICONDA_INSTALLER = ""

    if os_type == "Darwin":
        # macOS
        MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-MacOSX-x86_64.sh"
        MINICONDA_CHECKSUM = "a95f99c31ee1d2bf87e51546b9c71f5820b792e05b0d2f4a1bc4618478efce15"
        MINICONDA_INSTALLER = "Miniconda3-py310_24.5.0-0-MacOSX-x86_64.sh"
    elif os_type == "Linux":
        # Linux
        MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh"
        MINICONDA_CHECKSUM = "a95f99c31ee1d2bf87e51546b9c71f5820b792e05b0d2f4a1bc4618478efce15"
        MINICONDA_INSTALLER = "Miniconda3-py310_24.5.0-0-Linux-x86_64.sh"
    else:
        print(f"Unsupported operating system: {os_type}")
        sys.exit(1)

    # ===== 查找conda并检测是否损坏 =====
    CONDA_BIN = find_conda_binary(CONDA_ROOT_PREFIX)
    CONDA_EXISTS = False
    CONDA_BASE_CORRUPTED = False

    if CONDA_BIN and os.path.isfile(CONDA_BIN):
        CONDA_EXISTS = True
        print("Checking conda base Python installation...")
        if not check_conda_base_python(CONDA_ROOT_PREFIX):
            print("WARNING: Conda base Python installation appears corrupted.")
            print("This can happen if the installation was interrupted or files were corrupted.")
            print("Verification details:")
            python_bin = os.path.join(CONDA_ROOT_PREFIX, "bin", "python")
            print(f"  - Python binary: {python_bin}")
            if os.path.exists(python_bin):
                print("  - Python exists: YES")
                print("  - Testing basic Python functionality...")
                try:
                    run_command([python_bin, "-c", "import sys; print(f'Python {sys.version}')"], check=False, silent=True)
                except:
                    print("  - Python test: FAILED")
            else:
                print("  - Python exists: NO")
            CONDA_BASE_CORRUPTED = True
        else:
            print("Conda base Python installation verified.")

    # ===== 安装/重装Miniconda =====
    if not CONDA_EXISTS or CONDA_BASE_CORRUPTED:
        if CONDA_BASE_CORRUPTED:
            print("Removing corrupted conda installation...")
            if os.path.exists(CONDA_ROOT_PREFIX):
                shutil.rmtree(CONDA_ROOT_PREFIX)
            print("Conda installation removed. Will reinstall...")
        
        print(f"Downloading Miniconda from {MINICONDA_URL}")
        os.makedirs(INSTALL_DIR, exist_ok=True)
        installer_path = os.path.join(INSTALL_DIR, MINICONDA_INSTALLER)

        # 下载Miniconda安装包
        if not os.path.exists(installer_path):
            run_command(["curl", "-Lk", MINICONDA_URL, "-o", installer_path], silent=True)
        
        # 执行安装，复刻set +e / set -e，忽略python.app报错
        print(f"Installing Miniconda to {CONDA_ROOT_PREFIX}")
        result = subprocess.run(
            ["bash", installer_path, "-b", "-p", CONDA_ROOT_PREFIX, "-u"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
        )
        INSTALLER_EXIT_CODE = result.returncode

        # 文件系统同步等待
        print("Waiting for installation to complete...")
        time.sleep(5)

        # 重试查找conda，最多3次
        print("Verifying Miniconda installation...")
        CONDA_BIN = ""
        for i in range(3):
            CONDA_BIN = find_conda_binary(CONDA_ROOT_PREFIX)
            if CONDA_BIN and os.path.isfile(CONDA_BIN):
                break
            if i < 2:
                print(f"Conda binary not found, waiting and retrying... (attempt {i+1}/3)")
                time.sleep(2)

        # 找不到conda时的兜底查找与修复
        if not CONDA_BIN or not os.path.isfile(CONDA_BIN):
            print("WARNING: Conda binary not found in standard locations after installation.")
            print("This may happen if python.app installation failed (expected on macOS).")
            print("")
            print("Searching for conda in installation directory...")
            
            # 查找所有conda二进制文件
            found_condas = []
            for p in Path(CONDA_ROOT_PREFIX).rglob("conda"):
                if p.is_file() and os.access(p, os.X_OK) and "/bin/conda" in str(p):
                    found_condas.append(str(p))
            
            if not found_condas:
                print("ERROR: No conda binaries found anywhere.")
                print(f"Installer exit code: {INSTALLER_EXIT_CODE}")
                sys.exit(1)
            
            print("Found conda binaries:")
            for c in found_condas[:5]:
                print(c)
            print("")

            # 从pkgs找conda并修复
            pkgs_conda = ""
            for p in found_condas:
                if "/pkgs/" in p:
                    pkgs_conda = p
                    break
            
            if pkgs_conda:
                print(f"Found conda at: {pkgs_conda}")
                print("Attempting to complete installation using this conda...")
                os.makedirs(os.path.join(CONDA_ROOT_PREFIX, "bin"), exist_ok=True)
                os.makedirs(os.path.join(CONDA_ROOT_PREFIX, "condabin"), exist_ok=True)

                # 复制conda到标准路径
                try:
                    shutil.copy(pkgs_conda, os.path.join(CONDA_ROOT_PREFIX, "bin", "conda"))
                    os.chmod(os.path.join(CONDA_ROOT_PREFIX, "bin", "conda"), 0o755)
                    CONDA_BIN = os.path.join(CONDA_ROOT_PREFIX, "bin", "conda")
                    print(f"Copied conda to {CONDA_BIN}")
                except:
                    try:
                        shutil.copy(pkgs_conda, os.path.join(CONDA_ROOT_PREFIX, "condabin", "conda"))
                        os.chmod(os.path.join(CONDA_ROOT_PREFIX, "condabin", "conda"), 0o755)
                        CONDA_BIN = os.path.join(CONDA_ROOT_PREFIX, "condabin", "conda")
                        print(f"Copied conda to {CONDA_BIN}")
                    except:
                        CONDA_BIN = pkgs_conda
                        print(f"Using conda from pkgs directory directly: {CONDA_BIN}")

                # 校验conda是否可执行
                if CONDA_BIN and os.path.isfile(CONDA_BIN):
                    try:
                        run_command([CONDA_BIN, "--version"], capture_output=True, silent=True)
                        print(f"Conda is now available and functional at: {CONDA_BIN}")
                    except:
                        print("WARNING: Conda binary found but cannot execute. Re-running installer to repair...")
                        subprocess.run(
                            ["bash", installer_path, "-b", "-p", CONDA_ROOT_PREFIX, "-u"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
                        )
                        time.sleep(5)
                        CONDA_BIN = find_conda_binary(CONDA_ROOT_PREFIX)
                        if not CONDA_BIN:
                            print("ERROR: Could not repair conda installation.")
                            sys.exit(1)
            else:
                print("ERROR: Miniconda installation failed - conda binary not found.")
                sys.exit(1)

        # 最终校验conda版本
        print(f"Found conda at: {CONDA_BIN}")
        print("Miniconda version:")
        run_command([CONDA_BIN, "--version"], silent=True)

        # 校验base python
        print("Verifying conda base Python after installation...")
        if not check_conda_base_python(CONDA_ROOT_PREFIX):
            print("ERROR: Conda base Python is still corrupted after installation.")
            sys.exit(1)
        print("Conda base Python verified successfully.")

        # 删除安装包
        if os.path.exists(installer_path):
            os.remove(installer_path)

    # ===== 最终查找conda，确保存在 =====
    CONDA_BIN = find_conda_binary(CONDA_ROOT_PREFIX)
    if not CONDA_BIN or not os.path.isfile(CONDA_BIN):
        print("ERROR: Could not find conda binary")
        sys.exit(1)

    # ===== 创建conda环境 (如果不存在) =====
    ABUS_GENUINE_INSTALLED = True
    if not os.path.exists(INSTALL_ENV_DIR):
        ABUS_GENUINE_INSTALLED = False
        print("Creating conda environment...")
        run_command([CONDA_BIN, "create", "-y", "-k", "--prefix", INSTALL_ENV_DIR, "python=3.10"], silent=True)

    # 校验环境python是否存在
    if not os.path.exists(os.path.join(INSTALL_ENV_DIR, "bin", "python")):
        print("Conda environment is empty.")
        sys.exit(1)

    # ===== 环境隔离配置，复刻原脚本环境变量 =====
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONPATH"] = ""
    os.environ["PYTHONHOME"] = ""
    os.environ["CUDA_PATH"] = INSTALL_ENV_DIR
    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]

    # ===== 校验并执行python环境检查 =====
    print("Verifying Python installation...")
    PYTHON_BIN = os.path.join(INSTALL_ENV_DIR, "bin", "python")
    if os.path.exists(PYTHON_BIN):
        try:
            run_command([PYTHON_BIN, "-c", "import sys; import math; import os"], capture_output=True, silent=True)
            print("Python installation verified successfully.")
        except:
            print("Python installation appears incomplete or corrupted.")
            print("Removing corrupted environment and recreating...")
            shutil.rmtree(INSTALL_ENV_DIR)
            run_command([CONDA_BIN, "create", "-y", "-k", "--prefix", INSTALL_ENV_DIR, "python=3.10"], silent=True)
            run_command([PYTHON_BIN, "-c", "import sys; import math; import os"], capture_output=True, silent=True)
            print("Python installation verified successfully.")
    else:
        print("Python binary not found. This should not happen if environment was created correctly.")

    # ===== 安装依赖 + 执行核心程序 =====
    print(f"Miniconda location: {CONDA_ROOT_PREFIX}")
    os.chdir(SCRIPT_DIR)

    if not ABUS_GENUINE_INSTALLED:
        print("Installing huggingface-hub==0.27.1...")
        run_command([CONDA_BIN, "run", "--prefix", INSTALL_ENV_DIR, "python", "-m", "pip", "install", "huggingface-hub==0.27.1"], silent=True)

    # 设置日志级别并执行核心脚本
    os.environ["LOG_LEVEL"] = "DEBUG"
    print("\n=== Starting ABUS Voice Service ===")
    run_command([CONDA_BIN, "run", "--prefix", INSTALL_ENV_DIR, "python", "start-abus.py", "voice"])

if __name__ == "__main__":
    # 强制Python3运行
    if sys.version_info < (3, 0):
        print("Error: This script requires Python 3.x")
        sys.exit(1)
    main()