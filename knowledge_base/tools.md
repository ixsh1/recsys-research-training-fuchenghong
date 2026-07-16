# Tools

## Git

- Initialize repository: `git init`
- Rename branch: `git branch -M main`
- Add remote: `git remote add origin <url>`
- Push branch: `git push -u origin main`

## SSH & Remote Server

### SSH 连接

```bash
ssh username@host -p 22
```

### VS Code Remote SSH

- 安装扩展 `Remote - SSH`
- SSH config (`~/.ssh/config`):
  ```
  Host lab-server
      HostName 222.198.156.91
      User your_username
      Port 22
  ```
- 命令面板: `Remote-SSH: Connect to Host`

### Windows 端口测试

```powershell
Test-NetConnection host -Port 22
```

## Linux 常用命令

### 文件操作

- `mkdir -p path` — 创建目录（递归）
- `touch file` — 创建空文件
- `cat file` — 查看全部内容
- `head -n N file` — 查看前 N 行
- `tail -n N file` — 查看后 N 行
- `cp src dst` — 复制文件
- `mv src dst` — 移动/重命名文件
- `rm -i file` — 安全删除（确认提示）
- `find . -maxdepth 1 -type f -print` — 查找文件

### 文本处理

- `grep -n "pattern" file` — 搜索并显示行号
- `printf "text\n" > file` — 写入文本到文件
- `wc -l file` — 统计行数

### 系统信息

- `du -sh .` — 当前目录大小
- `df -h ~` — Home 目录所在磁盘
- `ps -u "$USER"` — 查看自己的进程
- `which cmd` — 命令路径
- `history | tail -n 20` — 最近命令历史

### Shell 脚本

```bash
printf '#!/usr/bin/env bash\necho "Hello"\n' > script.sh
chmod u+x script.sh
./script.sh
```

## Conda 环境管理

### 基本命令

- `conda create -n name python=3.10 -y` — 创建环境
- `conda activate name` — 激活环境
- `conda deactivate` — 退出环境
- `conda env list` — 列出所有环境
- `conda --version` — 查看版本

### 镜像配置

- 配置文件: `~/.condarc`
- `conda config --show-sources` — 查看配置来源
- `conda config --show channels` — 查看频道
- `conda clean -i` — 清除索引缓存
- 清华 TUNA 镜像:
  ```
  channels:
    - conda-forge
    - nodefaults
  custom_channels:
    conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  ```

## pip 镜像配置

- `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- `pip config list` — 查看配置
- `pip config debug` — 调试配置

## GPU 工具

### nvidia-smi

- `nvidia-smi` — 查看所有 GPU 状态、显存使用、进程
- `watch -n 1 nvidia-smi` — 每秒刷新 GPU 状态
- `CUDA_VISIBLE_DEVICES=0` — 限制程序只看到指定 GPU

### PyTorch CUDA 检查

```python
import torch
torch.version.cuda          # CUDA runtime 版本
torch.cuda.is_available()   # CUDA 是否可用
torch.cuda.get_device_name(0)  # GPU 名称
torch.cuda.device_count()   # 可见 GPU 数量
```

### 显存控制

- `torch.cuda.set_per_process_memory_fraction(ratio, device)` — 限制 PyTorch 缓存分配器
- `torch.cuda.max_memory_allocated()` — PyTorch 峰值已分配显存
- `torch.cuda.max_memory_reserved()` — PyTorch 峰值已保留显存
- `torch.cuda.reset_peak_memory_stats()` — 重置峰值统计

## Notes

- Miniconda 安装: `wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh`
- GPU 显存管理需要同时控制 PyTorch 分配器（`set_per_process_memory_fraction`）和实际进程显存（`nvidia-smi` 监控）
- `torch.cuda.empty_cache()` 只释放 PyTorch 缓存，不直接降低 nvidia-smi 显示值
