"""
OSS 相关工具函数

提供阿里云 OSS 的上传、下载、列举等功能。
"""

import logging
from pathlib import Path
from typing import List

import oss2
from alibabacloud_sts20150401 import models as sts_models
from alibabacloud_sts20150401.client import Client as StsClient
from alibabacloud_tea_openapi import models as open_api_models

from neuracle.utils.env import get_aliyun_config

logger = logging.getLogger(__name__)

DEFAULT_STS_TOKEN_DURATION_SECONDES = 20 * 60
DEFAULT_STS_ROLE_SESSION_NAME = "gecko_check"


def get_assume_role(
    duration_seconds: int = DEFAULT_STS_TOKEN_DURATION_SECONDES,
    role_session_name: str = DEFAULT_STS_ROLE_SESSION_NAME,
    show_progress: bool = False,
) -> sts_models.AssumeRoleResponse:
    """获取阿里云STS临时授权

    原理：
        通过阿里云 STS 服务获取临时访问凭证（AccessKeyId、AccessKeySecret、SecurityToken）。
        凭证有效期默认20分钟，角色会话名称用于阿里云日志统计。

    Parameters
    ----------
    duration_seconds : int, optional
        授权有效时间, by default 20 * 60
    role_session_name : str, optional
        授权名称，用于阿里云日志统计, by default "gecko_check"
    show_progress : bool, optional
        是否打印凭证信息, by default False

    Returns
    -------
    sts_models.AssumeRoleResponse
        授权角色信息，包含 access_key_id, access_key_secret, security_token
    """
    aliyun_config = get_aliyun_config()
    config = open_api_models.Config(
        access_key_id=aliyun_config['access_key_id'],
        access_key_secret=aliyun_config['access_key_secret'],
        endpoint=aliyun_config['sts_endpoint'],
    )
    client = StsClient(config)
    assume_role_request = sts_models.AssumeRoleRequest(
        duration_seconds=duration_seconds,
        role_arn=aliyun_config['sts_role_arn'],
        role_session_name=role_session_name,
    )
    assume_role = client.assume_role(assume_role_request)
    if show_progress:
        logger.info("access_key_id: %s", assume_role.body.credentials.access_key_id)
        logger.info("access_key_secret: %s", assume_role.body.credentials.access_key_secret)
        logger.info("security_token: %s", assume_role.body.credentials.security_token)
    return assume_role


def get_bucket() -> oss2.Bucket:
    """获取 OSS Bucket 实例

    原理：
        通过 STS 临时凭证创建 OSS Bucket 实例，用于后续的文件操作。

    Returns
    -------
    oss2.Bucket
        OSS Bucket 实例
    """
    assume_role = get_assume_role()
    aliyun_config = get_aliyun_config()
    auth = oss2.StsAuth(
        assume_role.body.credentials.access_key_id,
        assume_role.body.credentials.access_key_secret,
        assume_role.body.credentials.security_token,
    )
    bucket = oss2.Bucket(
        auth,
        aliyun_config['oss_endpoint'],
        aliyun_config['bucket_name'],
    )
    return bucket


def get_file_keys_from_oss(
    bucket: oss2.Bucket,
    oss_prefix: str = "",
    save_path: Path | None = None,
    show_progress: bool = False,
) -> List[str]:
    """获取 OSS Bucket 下对应前缀的所有文件路径

    原理：
        递归遍历指定前缀下的所有文件，返回文件路径列表。可选保存到本地文件。

    Parameters
    ----------
    bucket : oss2.Bucket
        OSS Bucket 实例
    oss_prefix : str, optional
        需要查看的 OSS 前缀路径, by default ""
    save_path : Path | None, optional
        保存路径, by default None
    show_progress : bool, optional
        是否打印获取到的文件信息, by default False

    Returns
    -------
    List[str]
        对应前缀的所有文件路径(key)列表
    """
    all_file_keys = []

    def recursive_list_files(prefix: str):
        """递归列举指定前缀下的所有文件"""
        for obj in oss2.ObjectIterator(bucket, prefix=prefix):
            if obj.is_prefix():
                recursive_list_files(obj.key)
            else:
                if show_progress:
                    logger.info("Listed file: %s", obj.key)
                all_file_keys.append(obj.key)

    recursive_list_files(oss_prefix)

    if save_path:
        with open(save_path, "w", encoding="utf8") as f:
            for path in all_file_keys:
                f.write(path + "\n")
    return all_file_keys


def download_folder_from_oss(
    bucket: oss2.Bucket, oss_prefix: str, local_dir: Path
) -> None:
    """下载 OSS 中某个前缀下的所有文件到本地

    原理：
        根据 OSS 前缀获取所有文件列表，逐一下载并保留原有目录结构。

    Parameters
    ----------
    bucket : oss2.Bucket
        OSS Bucket 实例
    oss_prefix : str
        需要下载的 OSS 前缀
    local_dir : Path
        保存的本地路径
    """
    target_file_keys = get_file_keys_from_oss(bucket, oss_prefix)
    for file_key in target_file_keys:
        oss_rel_path = Path(file_key).relative_to(oss_prefix)
        target_path = local_dir / oss_rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("正在下载文件: %s", file_key)
            bucket.get_object_to_file(file_key, target_path)
            logger.info("文件下载成功: %s", file_key)
        except oss2.exceptions.OssError as e:
            logger.error("OSS错误: %s [%s]", e.message, oss_rel_path)
        except Exception as e:
            logger.error("下载文件失败: %s [%s]", e, oss_rel_path)


def upload_bytes_to_oss(
    bucket: oss2.Bucket,
    data: bytes,
    oss_key: str
) -> None:
    """上传字节数据到 OSS

    原理：
        将内存中的字节数据直接上传到 OSS，支持任意格式内容。

    Parameters
    ----------
    bucket : oss2.Bucket
        OSS Bucket 实例
    data : bytes
        要上传的字节数据
    oss_key : str
        OSS 中的目标路径（包含文件名）
    """
    try:
        logger.info("正在上传文件: %s", oss_key)
        bucket.put_object(oss_key, data)
        logger.info("文件上传成功: %s", oss_key)
    except oss2.exceptions.OssError as e:
        logger.error("OSS错误: %s [%s]", e.message, oss_key)
    except Exception as e:
        logger.error("上传文件失败: %s [%s]", e, oss_key)


def download_bytes_from_oss(
    bucket: oss2.Bucket,
    oss_key: str,
) -> bytes:
    """从 OSS 下载字节数据到内存

    原理：
        将 OSS 中的文件内容读取为字节数据并返回，不落盘。

    Parameters
    ----------
    bucket : oss2.Bucket
        OSS Bucket 实例
    oss_key : str
        OSS 中的文件路径

    Returns
    -------
    bytes
        文件的字节数据
    """
    try:
        logger.info("正在下载文件: %s", oss_key)
        result = bucket.get_object(oss_key)
        data = result.read()
        logger.info("文件下载成功: %s", oss_key)
        return data
    except oss2.exceptions.OssError as e:
        logger.error("OSS错误: %s [%s]", e.message, oss_key)
        raise
    except Exception as e:
        logger.error("下载文件失败: %s [%s]", e, oss_key)
        raise


__all__ = [
    'get_assume_role',
    'get_bucket',
    'get_file_keys_from_oss',
    'download_folder_from_oss',
    'upload_bytes_to_oss',
    'download_bytes_from_oss',
]
