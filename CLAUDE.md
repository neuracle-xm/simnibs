# SimNIBS 开发环境说明

## Python 环境

本项目使用 conda 虚拟环境 `simnibs`。

**重要**: 在执行任何 Python 相关命令之前，必须先激活 `simnibs` 环境。

### 激活环境

```bash
eval "$(conda shell.bash hook)"
conda activate simnibs
```

### 验证环境

```bash
python --version
# 应该显示: Python 3.11.10

python -c "import simnibs; print('SimNIBS loaded successfully')"
```

### 示例：正确执行命令

```bash
# 错误方式 - 会使用错误的 Python
python script.py

# 正确方式 - 先激活环境
eval "$(conda shell.bash hook)"
conda activate simnibs
python script.py
```
