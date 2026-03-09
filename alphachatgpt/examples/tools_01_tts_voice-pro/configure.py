#!/usr/bin/env python3
import os
import sys
import subprocess
import platform

def run_command(cmd, check=True, capture_output=False, encoding="utf-8"):
    """
    执行系统命令，模拟shell的set -e 效果，命令失败则退出
    :param cmd: 执行的命令列表
    :param check: True=命令失败则抛出异常并退出，False=不校验执行结果
    :param capture_output: True=捕获输出内容，False=直接打印到控制台
    :return: 命令执行结果对象
    """
    try:
        if capture_output:
            result = subprocess.run(
                cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding=encoding
            )
        else:
            result = subprocess.run(cmd, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败: {' '.join(cmd)}")
        if capture_output:
            print(f"错误信息: {e.stderr}")
        sys.exit(1)

def check_command_exists(cmd):
    """检查系统中是否存在指定命令，模拟shell的 command -v xxx"""
    return subprocess.run(
        ["which", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).returncode == 0

def install_homebrew():
    """安装Homebrew（macOS专属）"""
    print("Installing Homebrew...")
    brew_install_cmd = [
        "/bin/bash", "-c", 
        "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ]
    run_command(brew_install_cmd)

def main():
    # 打印原脚本的头部信息，完全一致
    print("=========================================================================")
    print("")
    print("  ABUS Configure [Version 3.0]")
    print("  contact: abus.aikorea@gmail.com")
    print("")
    print("=========================================================================")
    print("")

    # ========== 1. 获取脚本所在目录并切换到该目录 (对应原脚本 SCRIPT_DIR 逻辑) ==========
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)

    # ========== 2. 检查是否为root权限，打印权限提示 (对应原脚本 EUID 判断) ==========
    if os.geteuid() != 0:
        print("This script may need administrator privileges for some operations.")
        print("You may be prompted for your password.")
        print("")

    # ========== 3. 检测操作系统并执行对应配置 ==========
    os_type = platform.system()
    
    if os_type == "Darwin":
        # macOS 系统 (对应原脚本 [[ "$OSTYPE" == "darwin"* ]])
        print("macOS detected")
        print("")

        # 检查并安装Homebrew
        if not check_command_exists("brew"):
            install_homebrew()

        # 检查并安装ffmpeg
        if not check_command_exists("ffmpeg"):
            print("Installing ffmpeg...")
            run_command(["brew", "install", "ffmpeg"])

        # 检查并安装git
        if not check_command_exists("git"):
            print("Installing git...")
            run_command(["brew", "install", "git"])

        print("macOS configuration complete.")
        print("")

    elif os_type == "Linux":
        # Linux 系统 (对应原脚本 [[ "$OSTYPE" == "linux-gnu"* ]])
        print("Linux detected")
        print("")
        pkg_installed = False

        # 检测Linux包管理器并安装依赖，和原脚本完全一致的优先级和包列表
        if check_command_exists("apt-get"):
            # Debian / Ubuntu
            print("Detected apt package manager")
            run_command(["sudo", "apt-get", "update"])
            run_command(["sudo", "apt-get", "install", "-y", "git", "ffmpeg", "build-essential"])
            pkg_installed = True

        elif check_command_exists("yum"):
            # RHEL / CentOS
            print("Detected yum package manager")
            run_command(["sudo", "yum", "install", "-y", "git", "ffmpeg", "gcc", "gcc-c++", "make"])
            pkg_installed = True

        elif check_command_exists("dnf"):
            # Fedora
            print("Detected dnf package manager")
            run_command(["sudo", "dnf", "install", "-y", "git", "ffmpeg", "gcc", "gcc-c++", "make"])
            pkg_installed = True

        elif check_command_exists("pacman"):
            # Arch Linux
            print("Detected pacman package manager")
            run_command(["sudo", "pacman", "-S", "--noconfirm", "git", "ffmpeg", "base-devel"])
            pkg_installed = True

        if not pkg_installed:
            print("Unsupported Linux distribution. Please install git and ffmpeg manually.")
            sys.exit(1)

        # 检查NVIDIA GPU (Linux专属，对应原脚本 nvidia-smi 检测)
        if check_command_exists("nvidia-smi"):
            print("NVIDIA GPU detected. Please ensure CUDA toolkit is installed if needed.")
            run_command(["nvidia-smi"])

        print("Linux configuration complete.")
        print("")

    else:
        # 不支持的系统，和原脚本一致的提示并退出
        print(f"Unsupported operating system: {os_type}")
        sys.exit(1)

    # 脚本执行完成，和原脚本一致的结尾提示
    print("ABUS configure.py finished.")
    print("")

if __name__ == "__main__":
    # 仅支持Python3运行
    if sys.version_info < (3, 0):
        print("Error: This script requires Python 3.x")
        sys.exit(1)
    main()