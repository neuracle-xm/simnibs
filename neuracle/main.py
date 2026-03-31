"""
SimNIBS RabbitMQ 服务主入口

启动方式: python -m neuracle.main

该服务接收来自后端的任务请求，执行 CHARM 头模生成、TI 正向仿真、
TI 逆向仿真等任务，并通过 RabbitMQ 实时上报进度。
"""

import json
import logging
import os
import shutil
import threading
import time
from functools import partial
from pathlib import Path
from queue import Queue
from typing import Any

from neuracle.charm import (
    create_mesh,
    create_surfaces,
    denoise_inputs,
    init_atlas,
    prepare_t1,
    prepare_t2,
    run_segmentation,
)
from neuracle.logger import setup_logging
from neuracle.oss_tool import (
    download_file_from_oss,
    download_folder_from_oss,
    get_bucket,
    upload_file_to_oss,
    upload_folder_to_oss,
)
from neuracle.rabbitmq.listener import RabbitMQListener
from neuracle.rabbitmq.message_builder import build_progress_message
from neuracle.rabbitmq.schemas import (
    ForwardParams,
    InverseParams,
    ModelParams,
)
from neuracle.rabbitmq.sender_thread import SenderThread
from neuracle.rabbitmq.validator import (
    ValidationError,
    validate_ack_test_params,
    validate_forward_params,
    validate_inverse_params,
    validate_model_params,
)
from neuracle.ti_optimize import (
    get_electrode_mapping,
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.ti_simulation import (
    calculate_ti,
    run_tdcs_simulation,
    setup_electrode_pair1,
    setup_electrode_pair2,
    setup_session,
)
from neuracle.utils import (
    N_WORKERS,
    NON_ROI_THRESHOLD,
    cond_dict_to_list,
    dict_to_ack_test_params,
    dict_to_forward_params,
    dict_to_inverse_params,
    dict_to_model_params,
    find_montage_file,
    get_standardized_roi_path,
    load_env,
)
from neuracle.utils.env import get_aliyun_config, get_rabbitmq_config
from neuracle.utils.ti_export_utils import export_ti_to_nifti

logger = logging.getLogger(__name__)

# DEBUG 模式：True 时跳过 output_dir 删除
DEBUG = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def _ack_in_consumer_thread(
    channel: Any,
    delivery_tag: int,
    task_id: str,
    msg_type: str,
    reason: str,
) -> None:
    """在 RabbitMQ 消费线程中实际执行 ack。"""
    try:
        if channel is None or not getattr(channel, "is_open", False):
            logger.warning(
                "ack_sent 跳过，channel 不可用: task_id=%s type=%s delivery_tag=%s reason=%s",
                task_id,
                msg_type,
                delivery_tag,
                reason,
            )
            return
        channel.basic_ack(delivery_tag=delivery_tag)
        logger.info(
            "ack_sent: task_id=%s type=%s delivery_tag=%s reason=%s",
            task_id,
            msg_type,
            delivery_tag,
            reason,
        )
    except Exception:
        logger.exception(
            "发送 ack 失败: task_id=%s type=%s delivery_tag=%s reason=%s",
            task_id,
            msg_type,
            delivery_tag,
            reason,
        )


def schedule_ack(
    channel: Any,
    delivery_tag: int,
    task_id: str,
    msg_type: str,
    reason: str,
) -> None:
    """从工作线程把 ack 请求投递回消费线程执行。"""
    connection = getattr(channel, "connection", None)
    if connection is None or not getattr(connection, "is_open", False):
        logger.warning(
            "ack_scheduled 跳过，connection 不可用: task_id=%s type=%s delivery_tag=%s reason=%s",
            task_id,
            msg_type,
            delivery_tag,
            reason,
        )
        return
    logger.info(
        "ack_scheduled: task_id=%s type=%s delivery_tag=%s reason=%s",
        task_id,
        msg_type,
        delivery_tag,
        reason,
    )
    connection.add_callback_threadsafe(
        partial(
            _ack_in_consumer_thread,
            channel,
            delivery_tag,
            task_id,
            msg_type,
            reason,
        )
    )


def reset_task_output_dir(output_dir: str) -> None:
    """删除同 task_id 的旧输出，避免重跑时被残留结果污染。"""
    if os.path.isdir(output_dir):
        logger.info("清理旧任务输出目录: %s", output_dir)
        shutil.rmtree(output_dir)


DEFAULT_DIR_PATH = "m2m_ernie"
DEFAULT_DTI_FILE_PATH = "DTI_coregT1_tensor.nii.gz"


def normalize_dir_path(dir_path: str) -> str:
    """规范化相对目录路径，禁止绝对路径和父级跳转。空字符串使用默认值 m2m_ernie。"""
    normalized = dir_path.replace("\\", "/").strip().strip("/")
    if not normalized:
        normalized = DEFAULT_DIR_PATH
        logger.info("dir_path 为空，使用默认值: %s", normalized)
    path_obj = Path(normalized)
    if path_obj.is_absolute() or ".." in path_obj.parts:
        raise ValueError(f"非法 dir_path: {dir_path}")
    return normalized


def get_subject_dir(dir_path: str) -> Path:
    return DATA_ROOT / Path(normalize_dir_path(dir_path))


def get_task_output_dir(dir_path: str, suffix: str, task_id: str | None = None) -> Path:
    normalized = normalize_dir_path(dir_path)
    if task_id:
        return DATA_ROOT / f"{normalized}_{suffix}_{task_id}"
    return DATA_ROOT / f"{normalized}_{suffix}"


def get_model_mesh_path(dir_path: str) -> Path:
    return get_subject_dir(dir_path) / "model.msh"


def resolve_local_dti_path(dir_path: str, dti_file_path: str | None) -> str | None:
    if not dti_file_path:
        dti_file_path = DEFAULT_DTI_FILE_PATH
        logger.info("DTI_file_path 为空，使用默认值: %s", dti_file_path)
    return str(get_subject_dir(dir_path) / dti_file_path)


def build_storage_key(path: str) -> str:
    """将逻辑 OSS key 映射到真实存储 key。"""
    bucket_target = get_aliyun_config().get("bucket_target", "").strip("/")
    logical_key = path.strip("/")
    if bucket_target:
        return f"{bucket_target}/{logical_key}"
    return logical_key


def ensure_data_root() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def ensure_subject_cache(bucket: Any, dir_path: str) -> Path:
    """确保本地 subject 缓存存在；不存在时从 OSS 按前缀下载。"""
    ensure_data_root()
    normalized = normalize_dir_path(dir_path)
    subject_dir = get_subject_dir(normalized)
    if subject_dir.is_dir():
        logger.info("命中本地缓存: %s", subject_dir)
        return subject_dir

    logger.info("未命中本地缓存，开始下载前缀: %s", normalized)
    subject_dir.mkdir(parents=True, exist_ok=True)
    download_folder_from_oss(
        bucket,
        build_storage_key(normalized),
        subject_dir,
    )
    return subject_dir


def download_input_file(bucket: Any, oss_key: str, local_path: Path) -> Path:
    """下载单个输入文件并返回本地路径。"""
    download_file_from_oss(bucket, build_storage_key(oss_key), local_path)
    return local_path


def upload_model_outputs(
    bucket: Any, dir_path: str, subject_dir: Path
) -> dict[str, str]:
    normalized = normalize_dir_path(dir_path)
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
    }
    for oss_key, local_file in file_mappings.items():
        if local_file.is_file():
            logger.info("上传文件到 OSS: %s -> %s", local_file, oss_key)
            upload_file_to_oss(bucket, local_file, build_storage_key(oss_key))
            uploaded[local_file.name] = oss_key

    return uploaded


def upload_task_result(bucket: Any, local_file: Path, oss_key: str) -> str:
    logger.info("上传任务结果到 OSS: %s -> %s", local_file, oss_key)
    upload_file_to_oss(bucket, local_file, build_storage_key(oss_key))
    return oss_key


def mask_rabbitmq_config(config: dict[str, Any]) -> dict[str, Any]:
    """隐藏敏感配置，避免密码写入日志。"""
    masked = dict(config)
    if masked.get("password"):
        masked["password"] = "***"
    return masked


def send_progress(
    message_queue: Queue,
    task_id: str,
    msg_type: str,
    progress_rate: int,
    error_message: str | None = None,
    result: dict | None = None,
) -> None:
    """
    发送进度消息到 simnibs_to_backend 队列

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    msg_type : str
        任务类型：model, forward, inverse
    progress_rate : int
        进度百分比 (0-100)
    error_message : str, optional
        错误消息，无错时为 None
    result : dict, optional
        结果数据
    """
    progress_msg = build_progress_message(
        id=task_id,
        msg_type=msg_type,
        progress_rate=progress_rate,
        message=error_message,
        result=result,
    )

    message_queue.put(progress_msg)
    logger.debug("发送进度消息: %s %d%%", task_id, progress_rate)


def handle_model_task(message_queue: Queue, task_id: str, params: ModelParams) -> None:
    """
    处理 CHARM 头模生成任务

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    params : ModelParams
        头模生成参数
    """
    bucket = get_bucket()
    subject_dir = get_subject_dir(params.dir_path)
    ensure_data_root()
    subject_dir.mkdir(parents=True, exist_ok=True)
    t1_local_path = download_input_file(
        bucket,
        params.T1_file_path,
        subject_dir / Path(params.T1_file_path).name,
    )
    t2_local_path = None
    if params.T2_file_path:
        t2_local_path = download_input_file(
            bucket,
            params.T2_file_path,
            subject_dir / Path(params.T2_file_path).name,
        )
    # 头模生成本身不需要DTI文件，但是要先下载下来，仿真时要用
    if params.DTI_file_path:
        download_input_file(
            bucket,
            params.DTI_file_path,
            subject_dir / Path(params.DTI_file_path).name,
        )

    # 0% - 任务开始
    send_progress(message_queue, task_id, "model", 0)

    # 10% - prepare_t1
    prepare_t1(str(subject_dir), str(t1_local_path))
    send_progress(message_queue, task_id, "model", 10)

    # 20% - prepare_t2
    if t2_local_path:
        try:
            prepare_t2(str(subject_dir), str(t2_local_path))
        except Exception:
            prepare_t2(str(subject_dir), str(t2_local_path), force_sform=True)
        send_progress(message_queue, task_id, "model", 20)

    # 35% - denoise
    denoise_inputs(str(subject_dir))
    send_progress(message_queue, task_id, "model", 35)

    # 50% - init_atlas
    init_atlas(str(subject_dir))
    send_progress(message_queue, task_id, "model", 50)

    # 70% - segment
    run_segmentation(str(subject_dir))
    send_progress(message_queue, task_id, "model", 70)

    # 85% - create_surfaces
    create_surfaces(str(subject_dir))
    send_progress(message_queue, task_id, "model", 85)

    # 100% - mesh
    create_mesh(str(subject_dir))
    upload_model_outputs(bucket, params.dir_path, subject_dir)
    msh_path = f"{normalize_dir_path(params.dir_path)}/model.msh"
    result = {"msh_file_path": msh_path}
    logger.debug("msh_file_path: %s", msh_path)
    send_progress(message_queue, task_id, "model", 100, result=result)


def handle_forward_task(
    message_queue: Queue, task_id: str, params: ForwardParams
) -> None:
    """
    处理 TI 正向仿真任务

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    params : ForwardParams
        正向仿真参数
    """
    bucket = get_bucket()
    subject_dir = ensure_subject_cache(bucket, params.dir_path)
    output_dir = get_task_output_dir(params.dir_path, "TI_simulation", task_id)
    anisotropy_type = "vn" if params.anisotropy else "scalar"
    reset_task_output_dir(str(output_dir))

    # 获取 EEG 电极帽 CSV 文件路径
    eeg_cap = find_montage_file(str(subject_dir), params.montage)
    mesh_path = get_model_mesh_path(params.dir_path)
    dti_path = resolve_local_dti_path(params.dir_path, params.DTI_file_path)

    # 从 ElectrodeWithCurrent 对象中提取电极名称和电流
    electrode_A_names = [e.name for e in params.electrode_A]
    electrode_A_currents = [e.current_mA for e in params.electrode_A]
    electrode_B_names = [e.name for e in params.electrode_B]
    electrode_B_currents = [e.current_mA for e in params.electrode_B]

    # 0% - 任务开始
    send_progress(message_queue, task_id, "forward", 0)

    # 10% - setup_session
    S = setup_session(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir),
        anisotropy_type=anisotropy_type,
        cond=cond_dict_to_list(params.conductivity_config),
        fname_tensor=dti_path,
        eeg_cap=eeg_cap,
    )
    send_progress(message_queue, task_id, "forward", 10)

    # 20% - setup_electrode_pair1
    setup_electrode_pair1(
        session=S,
        electrode_pair1=electrode_A_names,
        current1=electrode_A_currents,
    )
    send_progress(message_queue, task_id, "forward", 20)

    # 35% - setup_electrode_pair2
    setup_electrode_pair2(
        session=S,
        electrode_pair2=electrode_B_names,
        current2=electrode_B_currents,
    )
    send_progress(message_queue, task_id, "forward", 35)

    # 70% - run_tdcs
    mesh1_path, mesh2_path = run_tdcs_simulation(
        session=S,
        subject_dir=str(subject_dir),
        output_dir=str(output_dir),
        n_workers=N_WORKERS,
    )
    send_progress(message_queue, task_id, "forward", 70)

    # 85% - calculate_ti
    ti_mesh_path = calculate_ti(
        mesh1_path=mesh1_path,
        mesh2_path=mesh2_path,
        output_dir=str(output_dir),
    )
    send_progress(message_queue, task_id, "forward", 85)

    # 95% - export_ti_to_nifti
    ti_nifti_path = export_ti_to_nifti(
        msh_path=ti_mesh_path,
        output_dir=str(output_dir),
        reference=str(subject_dir / params.T1_file_path),
        field_name="max_TI",
    )
    send_progress(message_queue, task_id, "forward", 95)

    ti_file_key = upload_task_result(
        bucket,
        Path(ti_nifti_path),
        f"{normalize_dir_path(params.dir_path)}_TI_simulation_{task_id}/TI_max_TI.nii.gz",
    )
    if DEBUG:
        logger.info("DEBUG 模式，跳过删除 output_dir: %s", output_dir)
    else:
        shutil.rmtree(output_dir, ignore_errors=True)

    result = {"TI_file": ti_file_key}
    logger.debug("TI_file: %s", ti_file_key)
    send_progress(message_queue, task_id, "forward", 100, result=result)


def handle_inverse_task(
    message_queue: Queue, task_id: str, params: InverseParams
) -> None:
    """
    处理 TI 逆向仿真任务

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    params : InverseParams
        逆向仿真参数
    """
    bucket = get_bucket()
    subject_dir = ensure_subject_cache(bucket, params.dir_path)
    output_dir = get_task_output_dir(params.dir_path, "TI_optimization", task_id)
    anisotropy_type = "vn" if params.anisotropy else "scalar"
    reset_task_output_dir(str(output_dir))

    # 获取 EEG 电极帽 CSV 文件路径（用于电极映射）
    net_electrode_file = find_montage_file(str(subject_dir), params.montage)
    mesh_path = get_model_mesh_path(params.dir_path)
    dti_path = resolve_local_dti_path(params.dir_path, params.DTI_file_path)

    # 电极和 ROI 参数
    roi_center = None
    roi_radius = None
    roi_center_space = "subject"
    roi_mask_path = None
    roi_mask_space = None
    focality_threshold = params.target_threshold

    if params.roi_type == "atlas" and params.roi_param.atlas_param:
        roi_mask_path = str(
            get_standardized_roi_path(
                params.roi_param.atlas_param.name,
                params.roi_param.atlas_param.area,
            )
        )
        if not os.path.exists(roi_mask_path):
            raise FileNotFoundError(
                f"标准化 ROI 不存在: {roi_mask_path}。请先运行 atlas 标准化和 ROI 生成脚本。"
            )
        roi_mask_space = "mni"
    elif params.roi_type == "mni_pos" and params.roi_param.mni_param:
        roi_center = params.roi_param.mni_param.center
        roi_radius = params.roi_param.mni_param.radius
        roi_center_space = "mni"

    # 0% - 任务开始
    send_progress(message_queue, task_id, "inverse", 0)

    # 10% - init_optimization
    opt = init_optimization(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir),
        anisotropy_type=anisotropy_type,
        cond=cond_dict_to_list(params.conductivity_config),
        fname_tensor=dti_path,
    )
    send_progress(
        message_queue,
        task_id,
        "inverse",
        10,
    )

    # 20% - setup_goal
    setup_goal(
        opt=opt,
        goal="focality",
        focality_threshold=[focality_threshold, NON_ROI_THRESHOLD],
        net_electrode_file=net_electrode_file,
    )
    send_progress(message_queue, task_id, "inverse", 20)

    # 35% - setup_electrodes_and_roi
    setup_electrodes_and_roi(
        opt=opt,
        goal="focality",
        roi_center=roi_center,
        roi_radius=roi_radius,
        roi_center_space=roi_center_space,
        roi_mask_path=roi_mask_path,
        roi_mask_space=roi_mask_space,
        non_roi_center=None,
        non_roi_radius=None,
    )
    send_progress(message_queue, task_id, "inverse", 35)

    # 85% - run_optimization
    run_optimization(opt=opt, n_workers=N_WORKERS)
    send_progress(message_queue, task_id, "inverse", 85)

    # 90% - get_electrode_mapping
    electrode_A, electrode_B = get_electrode_mapping(output_dir=str(output_dir))
    send_progress(message_queue, task_id, "inverse", 90)

    # 95% - export_ti_to_nifti
    # 从 dir_path 中提取 modelid，格式如 m2m_ernie -> ernie
    modelid = params.dir_path.split("m2m_")[-1]
    msh_name = f"{modelid}_tes_mapped_opt_surface_mesh.msh"
    msh_path = str(output_dir / "mapped_electrodes_simulation" / msh_name)
    ti_nifti_path = export_ti_to_nifti(
        msh_path=msh_path,
        output_dir=str(output_dir),
        reference=str(subject_dir / params.T1_file_path),
        field_name="max_TI",
    )
    send_progress(message_queue, task_id, "inverse", 95)

    ti_file_key = upload_task_result(
        bucket,
        Path(ti_nifti_path),
        f"{normalize_dir_path(params.dir_path)}_TI_optimization_{task_id}/TI_max_TI.nii.gz",
    )
    if DEBUG:
        logger.info("DEBUG 模式，跳过删除 output_dir: %s", output_dir)
    else:
        shutil.rmtree(output_dir, ignore_errors=True)

    result = {
        "TI_file": ti_file_key,
        "electrode_A": electrode_A,
        "electrode_B": electrode_B,
    }
    logger.debug(
        "TI_file: %s, electrode_A: %s, electrode_B: %s",
        ti_file_key,
        electrode_A,
        electrode_B,
    )
    send_progress(message_queue, task_id, "inverse", 100, result=result)


def handle_ack_test_task(message_queue: Queue, task_id: str, params: Any) -> None:
    """处理 ack 时机验证任务。"""
    send_progress(message_queue, task_id, "ack_test", 0)
    logger.info(
        "worker_started: task_id=%s type=ack_test sleep_seconds=%.2f",
        task_id,
        params.sleep_seconds,
    )
    time.sleep(params.sleep_seconds)
    logger.info("worker_finished: task_id=%s type=ack_test", task_id)
    result = {"sleep_seconds": params.sleep_seconds}
    send_progress(message_queue, task_id, "ack_test", 100, result=result)


def execute_task(
    message_queue: Queue,
    task_id: str,
    msg_type: str,
    task_handler: Any,
    params: Any,
    channel: Any,
    delivery_tag: int,
) -> None:
    """在线程中执行耗时任务，避免阻塞 RabbitMQ 回调线程。"""
    ack_reason = "task_finished"
    try:
        task_handler(message_queue, task_id, params)
        logger.info("任务完成: %s", task_id)
    except ValidationError as e:
        ack_reason = "task_validation_error"
        logger.error("参数验证失败: %s - %s", task_id, e)
        send_progress(message_queue, task_id, msg_type, 0, str(e))
    except Exception as e:
        ack_reason = "task_exception"
        logger.exception("任务执行失败: %s - %s", task_id, e)
        send_progress(message_queue, task_id, msg_type, 0, str(e))
    finally:
        schedule_ack(channel, delivery_tag, task_id, msg_type, ack_reason)


def handle_message(
    channel: Any, method: Any, properties: Any, body: bytes, message_queue: Queue
) -> None:
    """
    处理接收到的任务消息

    根据消息 type 分发到对应的处理函数。

    Parameters
    ----------
    channel : pika.channel.Channel
        RabbitMQ 通道
    method : pika.spec.Basic.Deliver
        消息传递信息
    properties : pika.spec.BasicProperties
        消息属性
    body : bytes
        消息内容
    message_queue : Queue
        消息队列
    """
    task_id = ""
    msg_type = ""
    delivery_tag = method.delivery_tag

    try:
        data = json.loads(body)
        task_id = data["id"]
        msg_type = data["type"]
        params_data = data["params"]

        logger.info("收到任务: %s, 类型: %s", task_id, msg_type)

        if msg_type == "model":
            validate_model_params(params_data)
            params = dict_to_model_params(params_data, task_id)
            task_handler = handle_model_task

        elif msg_type == "forward":
            validate_forward_params(params_data)
            params = dict_to_forward_params(params_data, task_id)
            task_handler = handle_forward_task

        elif msg_type == "inverse":
            validate_inverse_params(params_data)
            params = dict_to_inverse_params(params_data, task_id)
            task_handler = handle_inverse_task

        elif msg_type == "ack_test":
            validate_ack_test_params(params_data)
            params = dict_to_ack_test_params(params_data, task_id)
            task_handler = handle_ack_test_task

        else:
            raise ValueError(f"未知任务类型: {msg_type}")

        worker = threading.Thread(
            target=execute_task,
            args=(
                message_queue,
                task_id,
                msg_type,
                task_handler,
                params,
                channel,
                delivery_tag,
            ),
            daemon=True,
            name=f"task-{msg_type}-{task_id}",
        )
        worker.start()
        logger.info(
            "任务已提交到工作线程: %s, 类型: %s, 线程: %s",
            task_id,
            msg_type,
            worker.name,
        )

    except ValidationError as e:
        logger.error("参数验证失败: %s - %s", task_id, e)
        send_progress(message_queue, task_id, msg_type, 0, str(e))
        _ack_in_consumer_thread(
            channel,
            delivery_tag,
            task_id,
            msg_type or "unknown",
            "sync_validation_error",
        )

    except Exception as e:
        logger.exception("任务执行失败: %s - %s", task_id, e)
        send_progress(message_queue, task_id, msg_type, 0, str(e))
        _ack_in_consumer_thread(
            channel, delivery_tag, task_id, msg_type or "unknown", "sync_exception"
        )


def run_service(config: dict) -> None:
    """
    启动 RabbitMQ 服务

    Parameters
    ----------
    config : dict
        RabbitMQ 配置字典
    """
    message_queue: Queue = Queue()

    # 启动发送线程
    sender = SenderThread(
        host=config["host"],
        port=config["port"],
        queue_name=config["send_queue_name"],
        message_queue=message_queue,
        username=config["username"],
        password=config["password"],
        virtual_host=config["virtual_host"],
        heartbeat=config["heartbeat"],
        blocked_connection_timeout=config["blocked_connection_timeout"],
        socket_timeout=config["socket_timeout"],
        connection_attempts=config["connection_attempts"],
        retry_delay=config["retry_delay"],
    )
    sender.start()

    # 启动监听
    listener = RabbitMQListener(
        host=config["host"],
        port=config["port"],
        queue_name=config["listen_queue_name"],
        username=config["username"],
        password=config["password"],
        virtual_host=config["virtual_host"],
        heartbeat=config["heartbeat"],
        blocked_connection_timeout=config["blocked_connection_timeout"],
        socket_timeout=config["socket_timeout"],
        connection_attempts=config["connection_attempts"],
        retry_delay=config["retry_delay"],
    )

    logger.info("SimNIBS RabbitMQ 服务已启动")
    logger.info("监听队列: %s", config["listen_queue_name"])
    logger.info("发送队列: %s", config["send_queue_name"])

    callback = partial(handle_message, message_queue=message_queue)
    try:
        listener.consume_forever(callback)
    finally:
        sender.stop()


def main() -> None:
    """主入口函数"""
    setup_logging()
    load_env()
    config = get_rabbitmq_config()
    logger.info("RabbitMQ 配置: %s", mask_rabbitmq_config(config))

    run_service(config)


if __name__ == "__main__":
    main()
