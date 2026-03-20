# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

SimNIBS 是一个用于计算经颅电刺激（TES）和经颅磁刺激（TMS）产生的电场的软件包。项目主要分为三个部分：

1. **分割与网格生成**（segmentation/）- MRI 图像自动分割和网格化
2. **有限元仿真**（simulation/）- FEM 电场计算
3. **优化**（optimization/）- TES/TMS 刺激参数优化

## Python 环境

**必须使用 conda 虚拟环境**。项目依赖特定的 CGAL、Boost 和其他编译库。

### 激活环境

```bash
eval "$(conda shell.bash hook)"
conda activate simnibs
```

## 常用命令

### CLI 工具

项目提供了大量命令行工具（定义在 `pyproject.toml` 的 `[project.scripts]`）：

- `charm` - 头部模型分割和网格生成
- `simnibs` / `run_simnibs` - 运行仿真
- `simnibs_gui` - 启动 GUI
- `charm_tms` - TMS 仿真
- `eeg_positions` - EEG 电极位置
- `optimize_tes` / `optimize_tms` - 刺激优化

## 核心架构

### 主要模块结构

**simnibs/segmentation/** - MRI 分割
- `charm_main.py` - CHARM 分割流程主入口
- `charm_utils.py` - 分割工具函数
- `brain_surface.py` - 大脑表面重建
- 依赖 CAT12 和 samseg 进行分割
- 包含 Cython 扩展：`_cat_c_utils`, `_marching_cubes_lewiner_cy`, `_thickness`

**simnibs/mesh_tools/** - 网格处理
- `mesh_io.py` - 网格文件 I/O（Msh, Nodes, Elements 类）
- `meshing.py` - 网格生成操作
- `cgal/` - CGAL C++ 扩展（需要 CGAL >= 5, Boost, Eigen3）
  - `create_mesh_surf.pyx` - 表面网格生成
  - `create_mesh_vol.pyx` - 体积网格生成
  - `polygon_mesh_processing.pyx` - 网格处理

**simnibs/simulation/** - FEM 仿真
- `sim_struct.py` - 仿真数据结构（SESSION, TDCS, TMSLEE, 等）
- `fem.py` - FEM 求解器主接口
- `onlinefem.py` - 在线 FEM 求解
- `electrode_placement.py` - 电极放置
- `biot_savart.py` - TMS 线圈磁场计算（毕奥-萨伐尔定律）
- `tms_coil/` - TMS 线圈模型和变形
- `pygpc/` - 多项式混沌展开（GPC）用于不确定性量化

**simnibs/optimization/** - 刺激优化
- `tdcs_optimization.py` - TDCS 优化
- `tms_optimization.py` - TMS 优化
- `tms_flex_optimization.py` - TMS 灵活优化
- `tes_flex_optimization/` - TES 灵活阵列优化
- `ADMlib.py` - 辅助偶极子方法（第三方，GPL-v2.0）

**simnibs/utils/** - 工具函数
- `file_finder.py` - SubjectFiles 类用于定位主题文件（m2m 文件夹结构）
- `transformations.py` - MNI<->主体坐标转换
- `region_of_interest.py` - 感兴趣区域（ROI）定义
- `cond_utils.py` - 组织电导率管理
- `nnav.py` - 神经导航系统接口

### 关键数据结构

**SubjectFiles**（`utils/file_finder.py`）
- 管理主题文件夹结构（`m2m_{subjid}/`）
- 定位网格、NIfTI、变换矩阵等文件

**Msh**（`mesh_tools/mesh_io.py`）
- 网格数据结构（节点、元素、字段数据）
- 支持多种网格格式（.msh, .vtk 等）

**SESSION** 及子类（`simulation/sim_struct.py`）
- `SESSION` - 仿真会话容器
- `TDCS` - TDCS 仿真设置
- `TMSLEADFIELD` / `TMSLEADFIELD_EEG` - TMS 仿真设置
- `poslist` - 仿真列表

## 重要注意事项

### C++ 扩展编译

项目包含多个 Cython/C++ 扩展，编译要求：
- **Windows**: MSVC >= 14.0, conda 安装 boost
- **Linux**: GCC >= 6.3, 系统 boost
- **macOS**: Apple Clang == 10.0.1, Homebrew boost

CGAL 扩展在 `setup.py` 中定义，编译时自动下载 CGAL（header-only）。

### 文件路径约定

主题文件夹结构：`m2m_{subjid}/`
- `{subjid}.msh` - 头部网格
- `T1fs.nii` - T1 图像
- `segm.nii` - 分割结果
- `wm.nii` - 白质
- `gm.nii` - 灰质
- `csf.nii` - 脑脊液
- 等等...

### 第三方代码

项目包含多个第三方组件（见 `3RD-PARTY.md`）：
- CGAL (GPLv3+)
- CAT12 (GPLv2+)
- ADMlib (GPLv2, 仅学术用途)
- PETSc (BSD)
- 等等

### 生成时的规范

1. 新代码都只生成在`neuracle`以及子文件夹中
2. 不要生成测试代码
3. docstring使用NumPy格式
4. 注释使用中文，专有名词使用英文
5. 函数参数需要标注类型，返回值也需要标注类型
6. 禁止在函数内部使用import，所有import都在模块头部
7. 生成代码时，在docstring中要写清楚原理和用法
8. 如果需要生成一些临时用的脚本/文件/工具之类，全部放在 `private_gitignore/` 下
9. 生成新代码或者修改现有代码时，记得更新根目录下的`PROJECT_STRUCTURE.md`，每一行的说明仍然要对齐
10. 生成代码前需要输出方案，放在`neuracle/docs`下，格式为markdown
11. 例子类的运行代码都放在`neuracle/demo`下
12. 函数内部不要留空白行
13. logger用模块级别的,不要放到类内部
14. 用logger = logging.getLogger(__name__)的方式，不要写具体的模块名字
15. 不要写单元测试
16. Use lazy % formatting in logging functions
17. 函数内部变量名使用snake_case