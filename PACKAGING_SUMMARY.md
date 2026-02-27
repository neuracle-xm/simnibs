# Charm.py PyInstaller 打包总结

## 构建成功

已成功使用 PyInstaller 将 SimNIBS 的 `charm.py` 打包为独立的 Windows 可执行文件。

## 生成的文件

### 主要文件
1. **charm.spec** - PyInstaller 规范文件
2. **pyi_rth_simnibs.py** - 运行时钩子（处理 petsc4py 依赖问题）
3. **build_charm.bat** - Windows 构建脚本

### 打包输出
- **位置**: `dist/charm/`
- **主可执行文件**: `dist/charm/charm.exe`
- **打包类型**: 目录模式（非 onefile）

## 使用方法

### 运行可执行文件
```batch
cd dist\charm
charm.exe --help
charm.exe --version
```

### 重新构建
```batch
# 方法 1: 使用批处理脚本
build_charm.bat

# 方法 2: 直接使用 PyInstaller
pyinstaller -y charm.spec
```

## 解决的问题

1. **petsc4py/mpi4py 依赖问题**
   - 问题：simnibs/__init__.py 导入了整个 simulation 模块，该模块依赖 petsc4py
   - 解决方案：创建运行时钩子 `pyi_rth_simnibs.py`，在导入前创建 dummy 模块

2. **数据文件路径**
   - 包含了所有必需的资源文件：
     - external/bin/win/ 中的 Windows 二进制文件
     - segmentation/atlases/ 图集数据
     - resources/ElectrodeCaps_MNI/ EEG 电极帽模板
     - resources/templates/ MNI 模板
     - resources/coil_models/ 线圈模型

3. **Cython 扩展**
   - 所有 .pyd 文件已正确包含

## 环境要求

- Python 3.11.10 (conda simnibs_env)
- PyInstaller 6.19.0
- 所有 SimNIBS 依赖包

## 注意事项

1. **排除的模块**：为了减小体积和避免问题，排除了以下模块：
   - petsc4py, mpi4py（不需要用于 charm）
   - PyQt5（GUI 相关）
   - jupyterlab, pytest, mock（开发工具）

2. **运行时依赖**：打包后的可执行文件需要：
   - Windows 运行时库（已包含）
   - 所有必需的 DLL 文件（已包含）

## 测试结果

- `--help` 命令：正常显示帮助信息
- `--version` 命令：正常显示版本号
- 可执行文件位于：`C:\Users\mdrs\Documents\simnibs\dist\charm\charm.exe`
