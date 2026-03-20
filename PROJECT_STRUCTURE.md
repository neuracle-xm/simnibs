# SimNIBS 项目结构

```
simnibs/
│
├── .azure-pipelines/                    # Azure Pipelines CI/CD 配置文件
├── .claude/                             # Claude Code 配置和记忆文件
├── .vscode/                             # VS Code 编辑器配置
├── build/                               # 构建输出目录（编译后的扩展模块）
├── dist/                                # 分发包输出目录
├── docs/                                # 项目文档（Sphinx）
│   ├── _build/                          # 文档构建输出
│   ├── _static/                         # 文档静态资源
│   ├── build/                           # 文档构建脚本
│   ├── data/                            # 文档数据文件
│   ├── documentation/                   # 核心文档内容
│   ├── images/                          # 文档图片资源
│   ├── installation/                    # 安装说明文档
│   └── tutorial/                        # 教程文档
├── neuracle/                            # Neuracle 定制化代码（新代码目录）
│   ├── .env                              # 环境变量配置文件
│   ├── __pycache__/                      # Python 字节码缓存
│   ├── demo/                             # 示例代码
│   │   ├── oss_example.py               # OSS 上传下载示例
│   │   ├── rabbitmq_example.py          # RabbitMQ 监听器使用示例
│   │   └── logger_example.py            # Logger 使用示例
│   ├── docs/                             # 方案文档
│   ├── env.py                            # 环境变量读取模块
│   ├── log/                              # 日志文件目录（自动创建）
│   │   ├── debug.log                     # DEBUG 级别日志
│   │   ├── info.log                      # INFO 级别日志
│   │   ├── warning.log                   # WARNING 级别日志
│   │   └── error.log                     # ERROR 级别日志
│   ├── logger/                           # 日志配置模块
│   │   └── __init__.py                  # 日志配置和导出
│   ├── oss_tool/                          # OSS 工具模块
│   │   └── __init__.py                  # OSS 上传下载工具函数
│   └── rabbitmq/                         # RabbitMQ 消息功能
│       ├── __init__.py                  # 包初始化文件，导出公共 API
│       ├── listener.py                  # RabbitMQ 监听器实现
│       └── sender.py                    # RabbitMQ 发送器实现
├── packing/                             # 安装包打包脚本
│   └── macOS_installer/                 # macOS 安装包构建脚本
├── private_gitignore/                   # 临时脚本/文件/工具（本地私有）
├── simnibs/                             # 主 Python 包
│   │
│   ├── GUI/                             # PyQt5 图形用户界面
│   │
│   ├── _internal_resources/             # 内部资源文件
│   │   ├── html/                        # HTML 模板文件
│   │   ├── icons/                       # 图标资源
│   │   └── testing_files/               # 测试用数据文件
│   │
│   ├── cli/                             # 命令行工具入口
│   │   ├── tests/                       # CLI 测试
│   │   └── utils/                       # CLI 工具函数
│   │
│   ├── eeg/                             # EEG 正向问题计算
│   │   └── tests/                       # EEG 模块测试
│   │
│   ├── examples/                        # 示例脚本和教程
│   │   ├── analysis/                    # 结果分析示例
│   │   ├── coils/                       # 线圈相关示例
│   │   ├── optimization/                # 优化示例
│   │   ├── simulations/                 # 仿真示例
│   │   ├── tes_flex_optimization/       # TES 灵活优化示例
│   │   ├── tests/                       # 示例测试
│   │   ├── tms_flex_optimization/       # TMS 灵活优化示例
│   │   ├── uncertainty_quantification/  # 不确定性量化示例
│   │   └── utilities/                   # 工具函数示例
│   │
│   ├── external/                        # 外部二进制依赖
│   │   ├── bin/                         # 平台特定二进制文件
│   │   └── wheels/                      # Python wheel 包
│   │
│   ├── matlab_tools/                    # MATLAB 工具箱
│   │   └── @gifti/                      # GIfTI 文件读写（MATLAB 类）
│   │
│   ├── mesh_tools/                      # 网格处理工具
│   │   ├── cgal/                        # CGAL C++ 扩展（网格生成）
│   │   └── tests/                       # 网格工具测试
│   │
│   ├── optimization/                    # 刺激参数优化
│   │   ├── tes_flex_optimization/       # TES 灵活阵列优化
│   │   └── tests/                       # 优化模块测试
│   │
│   ├── resources/                       # 资源文件
│   │   ├── ElectrodeCaps_MNI/           # MNI 空间 EEG 电极帽定义
│   │   ├── coil_models/                 # TMS 线圈模型库
│   │   └── templates/                   # 模板文件（MNI 等）
│   │
│   ├── segmentation/                    # MRI 分割和网格生成
│   │   ├── atlases/                     # 分割图谱
│   │   ├── cat_c_utils/                 # CAT12 C 工具
│   │   ├── simnibs_samseg/              # SamSeg 分割算法
│   │   └── tests/                       # 分割模块测试
│   │
│   ├── simulation/                      # FEM 仿真求解器
│   │   ├── pygpc/                       # 多项式混沌展开（GPC）
│   │   ├── tests/                       # 仿真模块测试
│   │   └── tms_coil/                    # TMS 线圈模型和变形
│   │
│   └── utils/                           # 通用工具函数
│       └── tests/                       # 工具模块测试
│
├── simnibs.egg-info/                    # Python 包元信息
├── 3RD-PARTY.md                         # 第三方组件许可证声明
├── CLAUDE.md                            # Claude Code 开发指南
├── LICENSE.txt                          # GPL v3 许可证
├── MANIFEST.in                          # 包含文件清单
├── README.md                            # 项目说明
├── environment_linux.yml                # Linux conda 环境配置
├── environment_macOS.yml                # macOS conda 环境配置
├── environment_win.yml                  # Windows conda 环境配置
├── pyproject.toml                       # 现代 Python 项目配置
├── setup.py                             # 包安装脚本（含 C++ 扩展）
└── vc140.pdb                            # MSVC 调试符号文件
```
