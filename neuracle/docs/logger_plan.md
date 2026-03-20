# 日志系统改造方案

## 需求分析

1. 创建通用的 neuracle 级别日志系统（不局限于 rabbitmq）
2. 将 logger 改为模块级别（非类内部）
3. 增加 logger 保存到文件功能
4. 日志路径可配置，默认为 `neuracle/log/`
5. 四个级别各自保存到不同文件：
   - `debug.log` - DEBUG 级别
   - `info.log` - INFO 级别
   - `warning.log` - WARNING 级别
   - `error.log` - ERROR 级别
6. 自动回滚：每个文件最大 1MB，最多保留 20 个备份文件

## 实现方案

### 1. 创建日志配置模块 `neuracle/logger/`

使用 Python 标准库的 `logging.handlers.RotatingFileHandler` 实现自动回滚功能。
模块导入时自动配置，之后使用标准 logging 库即可。

### 2. 日志配置策略

| 级别    | 文件名      | 过滤级别       |
| ------- | ----------- | -------------- |
| DEBUG   | debug.log   | DEBUG 及以上   |
| INFO    | info.log    | INFO 及以上    |
| WARNING | warning.log | WARNING 及以上 |
| ERROR   | error.log   | ERROR 及以上   |

### 3. 日志格式

```
%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s
```

## 使用方法

```python
# 方式一：使用默认配置（模块导入时自动配置）
import logging
import neuracle.logger  # 导入触发自动配置

logger = logging.getLogger('neuracle.mymodule')
logger.info("message: %s", var)

# 方式二：手动指定日志目录
import logging
from neuracle.logger import setup_logging

setup_logging(log_dir='/path/to/logs')  # 指定日志目录

logger = logging.getLogger('neuracle.mymodule')
logger.info("message: %s", var)
```

## 设计原理

使用 Python 标准 logging 模块的单例机制：
1. `neuracle.logger` 模块导入时自动调用 `setup_logging()` 使用默认路径配置
2. 可手动调用 `setup_logging(log_dir=xxx)` 指定日志目录
3. 所有以 'neuracle' 开头的子 logger 自动继承配置
4. 使用标准的 `logging.getLogger()` 获取 logger 实例
5. 不会创建重复的文件处理器

## 文件结构

```
neuracle/
├── logger/                        # 日志配置模块
│   └── __init__.py               # 日志配置（模块导入时自动执行）
├── rabbitmq/                      # RabbitMQ 消息监听功能
│   ├── __init__.py              # 包初始化文件
│   ├── config.py                # RabbitMQ 配置文件模块
│   └── listener.py              # RabbitMQ 监听器实现
├── demo/                         # 示例代码
│   ├── rabbitmq_example.py      # RabbitMQ 使用示例
│   └── logger_example.py        # Logger 使用示例
└── log/                          # 日志目录（默认，自动创建）
    ├── debug.log                 # DEBUG 级别日志
    ├── info.log                  # INFO 级别日志
    ├── warning.log               # WARNING 级别日志
    └── error.log                 # ERROR 级别日志
```
