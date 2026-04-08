"""
SimNIBS RabbitMQ 服务主入口

启动方式: python -m neuracle.main

该服务接收来自后端的任务请求，执行 CHARM 头模生成、TI 正向仿真、
TI 逆向仿真等任务，并通过 RabbitMQ 实时上报进度。

架构说明：
    1. RabbitMQ Listener 监听任务队列，收到消息后回调 handle_message
    2. handle_message 在回调线程中解析消息，启动工作线程执行 execute_task
    3. 工作线程中根据任务类型调用对应的 handle_xxx_task 函数
    4. 任务完成后通过 schedule_ack 将 ack 请求投递回消费线程执行
    5. SenderThread 负责将从消息队列中取出进度消息发送到 RabbitMQ

进度恢复机制：
    - Model 任务支持进度恢复：当消息 redelivered=True 时，从进度文件恢复
    - 进度文件存储在 subject_dir / ".progress.txt"
    - 任务正常完成后删除进度文件，异常时保留用于下次恢复
"""

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime
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
    download_folder_from_oss,
    download_input_file,
    upload_model_outputs,
    upload_task_result,
)
from neuracle.rabbitmq.listener import RabbitMQListener
from neuracle.rabbitmq.message_builder import build_progress_message
from neuracle.rabbitmq.progress import (
    ForwardProgress,
    InverseProgress,
    ModelProgress,
    load_progress,
    save_progress,
)
from neuracle.rabbitmq.schemas import (
    ForwardParams,
    InverseParams,
    ModelParams,
)
from neuracle.rabbitmq.sender_thread import SenderThread
from neuracle.rabbitmq.task_handlers import (
    DEBUG,
    ensure_data_root,
    get_model_mesh_path,
    get_subject_dir,
    get_task_output_dir,
    mask_rabbitmq_config,
    normalize_dir_path,
    reset_task_output_dir,
    schedule_ack,
)
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
from neuracle.utils.env import get_rabbitmq_config
from neuracle.utils.ti_export_utils import export_ti_to_nifti

logger = logging.getLogger(__name__)


def send_progress(
    message_queue: Queue,
    task_id: str,
    msg_type: str,
    progress_rate: int,
    error_message: str | None = None,
    result: dict | None = None,
) -> None:
    """发送进度消息到 simnibs_to_backend 队列"""
    progress_msg = build_progress_message(
        id=task_id,
        msg_type=msg_type,
        progress_rate=progress_rate,
        message=error_message,
        result=result,
    )
    message_queue.put(progress_msg)
    logger.debug("发送进度消息: %s %d%%", task_id, progress_rate)


def handle_model_task(
    message_queue: Queue, task_id: str, params: ModelParams, redelivered: bool = False
) -> None:
    """处理 CHARM 头模生成任务

    Parameters
    ----------
    message_queue : Queue
        消息队列，用于发送进度更新
    task_id : str
        任务 ID
    params : ModelParams
        头模生成参数
    redelivered : bool
        消息是否为 RabbitMQ 重发（支持进度恢复：当 redelivered=True 时从进度文件恢复）
    """
    logger.info(
        "收到 Model 任务: task_id=%s, dir_path=%s, T1_file=%s, T2_file=%s, DTI_file=%s",
        task_id,
        params.dir_path,
        params.T1_file_path,
        params.T2_file_path,
        params.DTI_file_path,
    )
    # 获取 subject 目录
    subject_dir = get_subject_dir(params.dir_path)
    ensure_data_root()
    subject_dir.mkdir(parents=True, exist_ok=True)

    # 进度文件处理：只有 redelivered 消息才从进度文件恢复
    progress_file = subject_dir / ".progress.txt"
    if redelivered:
        current_progress = load_progress(progress_file)
        logger.info("任务 %s 为重发消息，从进度 %d 恢复", task_id, current_progress)
    else:
        # 新任务，清除旧进度文件（如果存在）
        if progress_file.exists():
            progress_file.unlink()
        current_progress = 0
        logger.info("任务 %s 为新消息，从头开始执行", task_id)
        # 发送任务开始进度
        send_progress(message_queue, task_id, "model", ModelProgress.START)

    # 构建文件路径
    t1_local_path = subject_dir / Path(params.T1_file_path).name
    t2_local_path = (
        subject_dir / Path(params.T2_file_path).name if params.T2_file_path else None
    )

    # 下载文件（只在进度 < 10 时执行）
    if current_progress < ModelProgress.PREPARE_T1_DONE:
        t1_local_path = download_input_file(params.T1_file_path, t1_local_path)
        if params.T2_file_path:
            t2_local_path = download_input_file(params.T2_file_path, t2_local_path)
        if params.DTI_file_path:
            download_input_file(
                params.DTI_file_path,
                subject_dir / Path(params.DTI_file_path).name,
            )

    # 步骤1: prepare_t1
    if current_progress < ModelProgress.PREPARE_T1_DONE:
        prepare_t1(str(subject_dir), str(t1_local_path))
        save_progress(progress_file, ModelProgress.PREPARE_T1_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.PREPARE_T1_DONE)

    # 步骤2: prepare_t2
    if current_progress < ModelProgress.PREPARE_T2_DONE:
        if t2_local_path and t2_local_path.exists():
            try:
                prepare_t2(str(subject_dir), str(t2_local_path))
            except Exception:
                prepare_t2(str(subject_dir), str(t2_local_path), force_sform=True)
        save_progress(progress_file, ModelProgress.PREPARE_T2_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.PREPARE_T2_DONE)

    # 步骤3: denoise
    if current_progress < ModelProgress.DENOISE_DONE:
        denoise_inputs(str(subject_dir))
        save_progress(progress_file, ModelProgress.DENOISE_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.DENOISE_DONE)

    # 步骤4: init_atlas
    if current_progress < ModelProgress.INIT_ATLAS_DONE:
        init_atlas(str(subject_dir))
        save_progress(progress_file, ModelProgress.INIT_ATLAS_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.INIT_ATLAS_DONE)

    # 步骤5: run_segmentation
    if current_progress < ModelProgress.SEGMENTATION_DONE:
        run_segmentation(str(subject_dir))
        save_progress(progress_file, ModelProgress.SEGMENTATION_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.SEGMENTATION_DONE)

    # 步骤6: create_surfaces
    if current_progress < ModelProgress.SURFACES_DONE:
        create_surfaces(str(subject_dir))
        save_progress(progress_file, ModelProgress.SURFACES_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.SURFACES_DONE)

    # 步骤7: create_mesh + 上传
    if current_progress < ModelProgress.COMPLETED:
        create_mesh(str(subject_dir))
        normalized = normalize_dir_path(params.dir_path)
        upload_model_outputs(params.dir_path, subject_dir, normalized)
        save_progress(progress_file, ModelProgress.COMPLETED)
        progress_file.unlink()
        msh_path = f"{normalize_dir_path(params.dir_path)}/model.msh"
        result = {"msh_file_path": msh_path}
        send_progress(
            message_queue, task_id, "model", ModelProgress.COMPLETED, result=result
        )


def handle_forward_task(
    message_queue: Queue, task_id: str, params: ForwardParams, redelivered: bool = False
) -> None:
    """处理 TI 正向仿真任务

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    params : ForwardParams
        正向仿真参数
    redelivered : bool
        消息是否为 RabbitMQ 重发（当前未使用，Forward 任务暂不支持进度恢复）
    """
    logger.info(
        "收到 Forward 任务: task_id=%s, dir_path=%s, montage=%s, anisotropy=%s, "
        "electrode_A=%s, electrode_B=%s, conductivity=%s",
        task_id,
        params.dir_path,
        params.montage,
        params.anisotropy,
        params.electrode_A,
        params.electrode_B,
        params.conductivity_config,
    )
    # 发送任务开始进度
    send_progress(message_queue, task_id, "forward", ForwardProgress.START)

    # 获取 subject 目录（如果不存在则从 OSS 下载）
    subject_dir = get_subject_dir(params.dir_path)
    if not subject_dir.is_dir():
        ensure_data_root()
        subject_dir.mkdir(parents=True, exist_ok=True)
        download_folder_from_oss(normalize_dir_path(params.dir_path), subject_dir)

    # 创建输出目录
    output_dir = get_task_output_dir(params.dir_path, "TI_simulation", task_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    anisotropy_type = "vn" if params.anisotropy else "scalar"

    # 清理旧输出
    if not DEBUG:
        reset_task_output_dir(str(output_dir))

    # 准备参数
    eeg_cap = find_montage_file(str(subject_dir), params.montage)
    mesh_path = get_model_mesh_path(params.dir_path)
    electrode_A_names = [e.name for e in params.electrode_A]
    # mA转化为A
    electrode_A_currents = [e.current_mA / 1000 for e in params.electrode_A]
    electrode_B_names = [e.name for e in params.electrode_B]
    electrode_B_currents = [e.current_mA / 1000 for e in params.electrode_B]

    # 步骤1: setup_session
    S = setup_session(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir),
        anisotropy_type=anisotropy_type,
        cond=cond_dict_to_list(params.conductivity_config),
        fname_tensor=params.DTI_file_path,
        eeg_cap=eeg_cap,
    )
    send_progress(message_queue, task_id, "forward", ForwardProgress.SESSION_SETUP)

    # 步骤2: setup_electrode_pair1
    setup_electrode_pair1(
        session=S,
        electrode_pair1=electrode_A_names,
        current1=electrode_A_currents,
    )
    send_progress(
        message_queue, task_id, "forward", ForwardProgress.ELECTRODE_PAIR1_DONE
    )

    # 步骤3: setup_electrode_pair2
    setup_electrode_pair2(
        session=S,
        electrode_pair2=electrode_B_names,
        current2=electrode_B_currents,
    )
    send_progress(
        message_queue, task_id, "forward", ForwardProgress.ELECTRODE_PAIR2_DONE
    )

    # 步骤4: run_tdcs_simulation
    mesh1_path, mesh2_path = run_tdcs_simulation(
        session=S,
        subject_dir=str(subject_dir),
        output_dir=str(output_dir),
        n_workers=N_WORKERS,
    )
    send_progress(
        message_queue, task_id, "forward", ForwardProgress.TDCS_SIMULATION_DONE
    )

    # 步骤5: calculate_ti
    ti_mesh_path = calculate_ti(
        mesh1_path=mesh1_path,
        mesh2_path=mesh2_path,
        output_dir=str(output_dir),
    )
    send_progress(message_queue, task_id, "forward", ForwardProgress.TI_CALCULATED)

    # 步骤6: export_ti_to_nifti
    ti_nifti_path = export_ti_to_nifti(
        msh_path=ti_mesh_path,
        output_dir=str(output_dir),
        reference=str(subject_dir / Path(params.T1_file_path).name),
        field_name="max_TI",
    )
    send_progress(message_queue, task_id, "forward", ForwardProgress.NIFTI_EXPORTED)

    # 步骤7: 上传结果
    ti_file_key = upload_task_result(
        Path(ti_nifti_path),
        f"{normalize_dir_path(params.dir_path)}_TI_simulation_{task_id}/TI_max_TI.nii.gz",
    )
    if not DEBUG:
        shutil.rmtree(output_dir, ignore_errors=True)

    result = {"TI_file": ti_file_key}
    send_progress(
        message_queue, task_id, "forward", ForwardProgress.COMPLETED, result=result
    )


def handle_inverse_task(
    message_queue: Queue, task_id: str, params: InverseParams, redelivered: bool = False
) -> None:
    """处理 TI 逆向仿真任务

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    params : InverseParams
        逆向仿真参数
    redelivered : bool
        消息是否为 RabbitMQ 重发（当前未使用，Inverse 任务暂不支持进度恢复）
    """
    roi_info = (
        f"atlas={params.roi_param.atlas_param}"
        if params.roi_param.atlas_param
        else f"mni={params.roi_param.mni_param}"
        if params.roi_param.mni_param
        else "None"
    )
    logger.info(
        "收到 Inverse 任务: task_id=%s, dir_path=%s, montage=%s, anisotropy=%s, "
        "current_A=%s, current_B=%s, roi_type=%s, roi=%s, target_threshold=%s, conductivity=%s",
        task_id,
        params.dir_path,
        params.montage,
        params.anisotropy,
        params.current_A,
        params.current_B,
        params.roi_type,
        roi_info,
        params.target_threshold,
        params.conductivity_config,
    )
    # 发送任务开始进度
    send_progress(message_queue, task_id, "inverse", InverseProgress.START)

    # 获取 subject 目录
    subject_dir = get_subject_dir(params.dir_path)
    if not subject_dir.is_dir():
        ensure_data_root()
        subject_dir.mkdir(parents=True, exist_ok=True)
        download_folder_from_oss(normalize_dir_path(params.dir_path), subject_dir)

    # 创建输出目录
    output_dir = get_task_output_dir(params.dir_path, "TI_optimization", task_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    anisotropy_type = "vn" if params.anisotropy else "scalar"

    # 清理旧输出
    if not DEBUG:
        reset_task_output_dir(str(output_dir))

    # 准备参数
    net_electrode_file = find_montage_file(str(subject_dir), params.montage)
    mesh_path = get_model_mesh_path(params.dir_path)
    roi_center = None
    roi_radius = None
    roi_center_space = "subject"
    roi_mask_path = None
    roi_mask_space = None
    focality_threshold = params.target_threshold

    # ROI 参数解析
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

    # 步骤1: init_optimization
    opt = init_optimization(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir),
        anisotropy_type=anisotropy_type,
        cond=cond_dict_to_list(params.conductivity_config),
        fname_tensor=params.DTI_file_path,
    )
    send_progress(message_queue, task_id, "inverse", InverseProgress.OPTIMIZATION_INIT)

    # 步骤2: setup_goal
    setup_goal(
        opt=opt,
        goal="focality",
        focality_threshold=[focality_threshold, NON_ROI_THRESHOLD],
        net_electrode_file=net_electrode_file,
    )
    send_progress(message_queue, task_id, "inverse", InverseProgress.GOAL_SETUP)

    # 步骤3: setup_electrodes_and_roi
    setup_electrodes_and_roi(
        opt=opt,
        goal="focality",
        mesh_file_path=mesh_path,
        electrode_current1=[c / 1000 for c in params.current_A],
        electrode_current2=[c / 1000 for c in params.current_B],
        roi_center=roi_center,
        roi_radius=roi_radius,
        roi_center_space=roi_center_space,
        roi_mask_path=roi_mask_path,
        roi_mask_space=roi_mask_space,
    )
    send_progress(
        message_queue, task_id, "inverse", InverseProgress.ELECTRODES_ROI_SETUP
    )

    # 步骤4: run_optimization
    run_optimization(opt=opt, n_workers=N_WORKERS)
    send_progress(message_queue, task_id, "inverse", InverseProgress.OPTIMIZATION_DONE)

    # 步骤5: get_electrode_mapping
    electrode_A, electrode_B = get_electrode_mapping(output_dir=str(output_dir))
    send_progress(
        message_queue, task_id, "inverse", InverseProgress.ELECTRODE_MAPPING_DONE
    )

    # 步骤6: export_ti_to_nifti
    msh_name = "model_tes_mapped_opt_head_mesh.msh"
    msh_path = str(output_dir / "mapped_electrodes_simulation" / msh_name)
    ti_nifti_path = export_ti_to_nifti(
        msh_path=msh_path,
        output_dir=str(output_dir),
        reference=str(subject_dir / Path(params.T1_file_path).name),
        field_name="max_TI",
    )
    send_progress(message_queue, task_id, "inverse", InverseProgress.NIFTI_EXPORTED)

    # 步骤7: 上传结果
    ti_file_key = upload_task_result(
        Path(ti_nifti_path),
        f"{normalize_dir_path(params.dir_path)}_TI_optimization_{task_id}/TI_max_TI.nii.gz",
    )
    if not DEBUG:
        shutil.rmtree(output_dir, ignore_errors=True)

    result = {
        "TI_file": ti_file_key,
        "electrode_A": electrode_A,
        "electrode_B": electrode_B,
    }
    send_progress(
        message_queue, task_id, "inverse", InverseProgress.COMPLETED, result=result
    )


def handle_ack_test_task(message_queue: Queue, task_id: str, params: Any) -> None:
    """处理 ack 时机验证任务（用于测试 RabbitMQ ack 机制）"""
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
    redelivered: bool = False,
) -> None:
    """在工作线程中执行耗时任务，避免阻塞 RabbitMQ 回调线程

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    msg_type : str
        任务类型
    task_handler : callable
        任务处理函数
    params : Any
        任务参数
    channel : Any
        RabbitMQ 通道
    delivery_tag : int
        消息 delivery tag
    redelivered : bool
        消息是否为重发
    """
    ack_reason = "task_finished"
    try:
        task_handler(message_queue, task_id, params, redelivered)
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
    """处理接收到的任务消息（RabbitMQ 回调函数）

    根据消息 type 分发到对应的处理函数。

    Parameters
    ----------
    channel : pika.channel.Channel
        RabbitMQ 通道
    method : pika.spec.Basic.Deliver
        消息传递信息（包含 delivery_tag 和 redelivered）
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
    redelivered = method.redelivered

    try:
        data = json.loads(body)
        task_id = data["id"]
        msg_type = data["type"]
        params_data = data["params"]

        logger.info(
            "收到任务: %s, 类型: %s, redelivered: %s", task_id, msg_type, redelivered
        )

        # 根据任务类型选择处理函数
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

        # 启动工作线程执行任务
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
                redelivered,
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
        channel.basic_ack(delivery_tag=delivery_tag)

    except Exception as e:
        logger.exception("任务执行失败: %s - %s", task_id, e)
        send_progress(message_queue, task_id, msg_type, 0, str(e))
        channel.basic_ack(delivery_tag=delivery_tag)


def run_service(config: dict) -> None:
    """启动 RabbitMQ 服务

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
    # 不同的 worker 使用不同的日志目录，防止多进程问题
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "log", f"worker_{timestamp}"
    )
    setup_logging(log_dir)
    global logger
    logger = logging.getLogger("neuracle.main")  # 使用固定名称，不要用 __name__
    load_env()
    config = get_rabbitmq_config()
    logger.info("RabbitMQ 配置: %s", mask_rabbitmq_config(config))

    run_service(config)


if __name__ == "__main__":
    main()
