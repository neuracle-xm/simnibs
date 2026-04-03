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

DEFAULT_STS_TOKEN_DURATION_SECONDES = 3600
DEFAULT_STS_ROLE_SESSION_NAME = "simnibs_session"


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
        授权有效时间, by default 3600
    role_session_name : str, optional
        授权名称，用于阿里云日志统计, by default ""
    show_progress : bool, optional
        是否打印凭证信息, by default False

    Returns
    -------
    sts_models.AssumeRoleResponse
        授权角色信息，包含 access_key_id, access_key_secret, security_token
    """
    aliyun_config = get_aliyun_config()
    config = open_api_models.Config(
        access_key_id=aliyun_config["access_key_id"],
        access_key_secret=aliyun_config["access_key_secret"],
        endpoint=aliyun_config["sts_endpoint"],
    )
    client = StsClient(config)
    assume_role_request = sts_models.AssumeRoleRequest(
        duration_seconds=duration_seconds,
        role_arn=aliyun_config["sts_role_arn"],
        role_session_name=role_session_name,
    )
    assume_role = client.assume_role(assume_role_request)
    if show_progress:
        logger.info("access_key_id: %s", assume_role.body.credentials.access_key_id)
        logger.info(
            "access_key_secret: %s", assume_role.body.credentials.access_key_secret
        )
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
        aliyun_config["oss_endpoint"],
        aliyun_config["bucket_name"],
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


def download_folder_from_oss(oss_prefix: str, local_dir: Path) -> None:
    """下载 OSS 中某个前缀下的所有文件到本地

    原理：
        根据 OSS 前缀获取所有文件列表，逐一下载并保留原有目录结构。
        每次下载前重新获取 STS Token，避免 Token 过期。

    Parameters
    ----------
    oss_prefix : str
        需要下载的 OSS 前缀
    local_dir : Path
        保存的本地路径

    Raises
    ------
    oss2.exceptions.OssError
        OSS 操作失败时抛出
    Exception
        其他下载错误时抛出
    """
    bucket = get_bucket()
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
            raise
        except Exception as e:
            logger.error("下载文件失败: %s [%s]", e, oss_rel_path)
            raise


def download_file_from_oss(
    oss_key: str,
    local_path: Path,
) -> None:
    """下载单个 OSS 文件到本地

    注意：每次下载前重新获取 STS Token，避免 Token 过期。
    """
    bucket = get_bucket()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("正在下载文件: %s -> %s", oss_key, local_path)
        bucket.get_object_to_file(oss_key, str(local_path))
        logger.info("文件下载成功: %s", oss_key)
    except oss2.exceptions.OssError as e:
        logger.error("OSS错误: %s [%s]", e.message, oss_key)
        raise
    except Exception as e:
        logger.error("下载文件失败: %s [%s]", e, oss_key)
        raise


def upload_file_to_oss(
    bucket: oss2.Bucket,
    local_path: Path,
    oss_key: str,
) -> None:
    """上传单个本地文件到 OSS。"""
    if not local_path.is_file():
        logger.error("本地文件不存在: %s", local_path)
        raise FileNotFoundError(f"本地文件不存在: {local_path}")
    try:
        logger.info("正在上传文件: %s -> %s", local_path, oss_key)
        bucket.put_object_from_file(oss_key, str(local_path))
        logger.info("文件上传成功: %s", oss_key)
    except oss2.exceptions.OssError as e:
        logger.error("OSS错误: %s [%s]", e.message, oss_key)
        raise
    except Exception as e:
        logger.error("上传文件失败: %s [%s]", e, oss_key)
        raise


def upload_folder_to_oss(
    bucket: oss2.Bucket,
    local_dir: Path,
    oss_prefix: str,
) -> list[str]:
    """递归上传本地目录到 OSS 前缀。"""
    if not local_dir.is_dir():
        logger.error("本地目录不存在: %s", local_dir)
        raise FileNotFoundError(f"本地目录不存在: {local_dir}")

    uploaded_keys: list[str] = []
    normalized_prefix = oss_prefix.strip("/")
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(local_dir).as_posix()
        oss_key = f"{normalized_prefix}/{rel_path}" if normalized_prefix else rel_path
        upload_file_to_oss(bucket, file_path, oss_key)
        uploaded_keys.append(oss_key)

    return uploaded_keys


def upload_bytes_to_oss(bucket: oss2.Bucket, data: bytes, oss_key: str) -> None:
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
        raise
    except Exception as e:
        logger.error("上传文件失败: %s [%s]", e, oss_key)
        raise


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


def build_storage_key(path: str) -> str:
    """将逻辑 OSS key 映射到真实存储 key

    原理：
        统一处理 OSS 路径格式，去除多余的前导斜杠，保持存储键一致性。

    Parameters
    ----------
    path : str
        原始 OSS 路径

    Returns
    -------
    str
        标准化后的 OSS 存储 key
    """
    logical_key = path.strip("/")
    return logical_key


def download_input_file(oss_key: str, local_path: Path) -> Path:
    """下载单个输入文件并返回本地路径

    原理：
        将 OSS 中的输入文件下载到本地指定路径，返回本地文件路径。

    Parameters
    ----------
    oss_key : str
        OSS 中的文件路径
    local_path : Path
        本地保存路径

    Returns
    -------
    Path
        本地文件路径

    Raises
    ------
    FileNotFoundError
        下载失败时抛出
    """
    download_file_from_oss(build_storage_key(oss_key), local_path)
    return local_path


def upload_model_outputs(
    dir_path: str, subject_dir: Path, normalized: str
) -> dict[str, str]:
    """上传模型输出到 OSS

    注意：每次上传前重新获取 STS Token，避免 Token 过期导致上传失败。
    """
    bucket = get_bucket()
    uploaded: dict[str, str] = {}
    folder_mappings = {
        f"{normalized}/eeg_positions": subject_dir / "eeg_positions",
        f"{normalized}/label_prep": subject_dir / "label_prep",
        f"{normalized}/surfaces": subject_dir / "surfaces",
        f"{normalized}/toMNI": subject_dir / "toMNI",
    }
    for oss_prefix, local_dir in folder_mappings.items():
        if local_dir.is_dir():
            logger.info("上传目录到 OSS: %s -> %s", local_dir, oss_prefix)
            upload_folder_to_oss(bucket, local_dir, build_storage_key(oss_prefix))
            uploaded[local_dir.name] = f"{oss_prefix}/"

    file_mappings = {
        f"{normalized}/model.msh": subject_dir / "model.msh",
        f"{normalized}/model.msh.opt": subject_dir / "model.msh.opt",
        f"{normalized}/T2_reg.nii.gz": subject_dir / "T2_reg.nii.gz",
    }
    for oss_key, local_file in file_mappings.items():
        if local_file.is_file():
            logger.info("上传文件到 OSS: %s -> %s", local_file, oss_key)
            upload_file_to_oss(bucket, local_file, build_storage_key(oss_key))
            uploaded[local_file.name] = oss_key

    return uploaded


def upload_task_result(local_file: Path, oss_key: str) -> str:
    """上传任务结果到 OSS

    注意：每次上传前重新获取 STS Token，避免 Token 过期导致上传失败。
    """
    bucket = get_bucket()
    logger.info("上传任务结果到 OSS: %s -> %s", local_file, oss_key)
    upload_file_to_oss(bucket, local_file, build_storage_key(oss_key))
    return oss_key


__all__ = [
    "get_assume_role",
    "get_bucket",
    "get_file_keys_from_oss",
    "download_file_from_oss",
    "download_folder_from_oss",
    "upload_file_to_oss",
    "upload_folder_to_oss",
    "upload_bytes_to_oss",
    "download_bytes_from_oss",
    "download_input_file",
    "build_storage_key",
    "upload_model_outputs",
    "upload_task_result",
]
