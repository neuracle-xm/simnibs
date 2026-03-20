"""
环境变量配置读取模块

提供从 .env 文件读取配置的函数。
"""

import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv

# 项目根目录
_PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
# .env 文件路径
_ENV_FILE: Final[Path] = _PROJECT_ROOT / 'neuracle' / '.env'


def load_env():
    """
    加载 .env 文件

    原理：
        使用 python-dotenv 库加载 .env 文件中的环境变量到 os.environ。
        如果 .env 文件不存在，静默失败。
    """
    load_dotenv(_ENV_FILE)


def get_rabbitmq_config() -> dict:
    """
    获取 RabbitMQ 配置

    原理：
        从环境变量中读取 RabbitMQ 相关配置，如果环境变量不存在则使用默认值。

    Returns:
        dict: 包含 host、port、queue_name 的配置字典
    """
    return {
        'host': os.getenv('RABBITMQ_HOST', 'localhost'),
        'port': int(os.getenv('RABBITMQ_PORT', '5672')),
        'listen_queue_name': os.getenv('RABBITMQ_LISTEN_QUEUE_NAME', ''),
        'send_queue_name': os.getenv('RABBITMQ_SEND_QUEUE_NAME', '')
    }


def get_aliyun_config() -> dict:
    """
    获取阿里云 OSS 配置

    原理：
        从环境变量中读取阿里云 OSS 相关配置，如果环境变量不存在则使用默认值。

    Returns:
        dict: 包含 access_key_id、access_key_secret、sts_role_arn、sts_endpoint、
              oss_endpoint、bucket_name、bucket_target 的配置字典
    """
    return {
        'access_key_id': os.getenv('ALI_ACCESS_KEY_ID', ''),
        'access_key_secret': os.getenv('ALI_ACCESS_KEY_SECRET', ''),
        'sts_role_arn': os.getenv('ALI_STS_ROLE_ARN', ''),
        'sts_endpoint': os.getenv('ALI_STS_ENDPOINT', 'sts.cn-shanghai.aliyuncs.com'),
        'oss_endpoint': os.getenv('ALI_OSS_ENDPOINT', 'https://oss-cn-shanghai.aliyuncs.com'),
        'bucket_name': os.getenv('ALI_BUCKET_NAME', ''),
        'bucket_target': os.getenv('ALI_BUCKET_TARGET', ''),
    }
