# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python 环境

**必须使用 conda 虚拟环境**。项目依赖特定的 CGAL、Boost 和其他编译库。

### 激活环境

```bash
eval "$(conda shell.bash hook)"
conda activate simnibs
```

### 生成时的规范

1. 执行python命令前必须激活虚拟环境
2. 新代码都只生成在`neuracle`以及子文件夹中
3. 不要生成测试代码
4. docstring使用NumPy格式
5. 注释使用中文，专有名词使用英文
6. 函数参数需要标注类型，返回值也需要标注类型
7. 禁止在函数内部使用import，所有import都在模块头部
8. 生成代码时，在docstring中要写清楚原理和用法
9. 如果需要生成一些临时用的脚本/文件/工具之类，全部放在 `private_gitignore/` 下
10. 生成新代码或者修改现有代码时，记得更新根目录下的`PROJECT_STRUCTURE.md`，每一行的说明仍然要对齐
11. 生成代码前需要输出方案，放在`neuracle/docs`下，格式为markdown
12. 例子类的运行代码都放在`neuracle/demo`下
13. 函数内部不要留空白行
14. logger用模块级别的,不要放到类内部
15. 用logger = logging.getLogger(__name__)的方式，不要写具体的模块名字
16. 不要写单元测试
17. Use lazy % formatting in logging functions
18. 函数内部变量名使用snake_case
19. 日志都使用中文
20. 函数内部变量名使用snake_case
