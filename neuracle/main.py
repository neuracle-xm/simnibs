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
    export_mz3 as optimize_export_mz3,
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
from neuracle.ti_simulation import (
    export_mz3 as sim_export_mz3,
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

logger = logging.getLogger(__name__)


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
    message: str | None = None,
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
    message : str, optional
        状态描述
    result : dict, optional
        结果数据
    """
    progress_msg = build_progress_message(
        id=task_id,
        msg_type=msg_type,
        progress_rate=progress_rate,
        message=message,
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
    # 0% - 任务开始
    send_progress(message_queue, task_id, "model", 0, "任务开始")

    # 10% - prepare_t1
    prepare_t1(params.dir_path, params.T1_file_path)
    send_progress(message_queue, task_id, "model", 10, "已完成: T1 图像准备与格式转换")

    # 20% - prepare_t2
    if params.T2_file_path:
        try:
            prepare_t2(params.dir_path, params.T2_file_path)
        except Exception:
            prepare_t2(params.dir_path, params.T2_file_path, force_sform=True)
        send_progress(message_queue, task_id, "model", 20, "已完成: T2 图像配准与准备")

    # 35% - denoise
    denoise_inputs(params.dir_path)
    send_progress(message_queue, task_id, "model", 35, "已完成: 输入图像降噪")

    # 50% - init_atlas
    init_atlas(params.dir_path)
    send_progress(
        message_queue, task_id, "model", 50, "已完成: Atlas 初始仿射配准与颈部校正"
    )

    # 70% - segment
    run_segmentation(params.dir_path)
    send_progress(message_queue, task_id, "model", 70, "已完成: 体积与表面分割")

    # 85% - create_surfaces
    create_surfaces(params.dir_path)
    send_progress(message_queue, task_id, "model", 85, "已完成: 皮层表面重建")

    # 100% - mesh
    create_mesh(params.dir_path)
    msh_path = os.path.join(params.dir_path, "model.msh")
    result = {"msh_file_path": msh_path}
    logger.debug("msh_file_path: %s", msh_path)
    send_progress(
        message_queue, task_id, "model", 100, "已完成: 四面体网格生成", result
    )


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
    output_dir = os.path.join(params.dir_path, "TI_simulation", task_id)
    anisotropy_type = "vn" if params.anisotropy else "scalar"
    reset_task_output_dir(output_dir)

    # 获取 EEG 电极帽 CSV 文件路径
    eeg_cap = find_montage_file(params.dir_path, params.montage)

    # 0% - 任务开始
    send_progress(message_queue, task_id, "forward", 0, "任务开始")

    # 10% - setup_session
    S = setup_session(
        subject_dir=params.dir_path,
        msh_file_path=params.msh_file_path,
        output_dir=output_dir,
        anisotropy_type=anisotropy_type,
        cond=cond_dict_to_list(params.cond),
        fname_tensor=params.DTI_file_path,
        eeg_cap=eeg_cap,
    )
    send_progress(message_queue, task_id, "forward", 10, "已完成: 配置会话参数")

    # 20% - setup_electrode_pair1
    setup_electrode_pair1(
        session=S,
        electrode_pair1=params.electrode_A,
        current1=params.current_A,
    )
    send_progress(message_queue, task_id, "forward", 20, "已完成: 配置第一个电极对")

    # 35% - setup_electrode_pair2
    setup_electrode_pair2(
        session=S,
        electrode_pair2=params.electrode_B,
        current2=params.current_B,
    )
    send_progress(message_queue, task_id, "forward", 35, "已完成: 配置第二个电极对")

    # 70% - run_tdcs
    mesh1_path, mesh2_path = run_tdcs_simulation(
        session=S,
        subject_dir=params.dir_path,
        output_dir=output_dir,
        n_workers=N_WORKERS,
    )
    send_progress(message_queue, task_id, "forward", 70, "已完成: TDCS 仿真计算")

    # 85% - calculate_ti
    ti_mesh_path = calculate_ti(
        mesh1_path=mesh1_path,
        mesh2_path=mesh2_path,
        output_dir=output_dir,
    )
    send_progress(message_queue, task_id, "forward", 85, "已完成: TI 场计算")

    # 95% - export_mz3
    mz3_path = sim_export_mz3(
        ti_mesh_path=ti_mesh_path,
        output_dir=output_dir,
        surface_type="central",
    )
    send_progress(message_queue, task_id, "forward", 95, "已完成: 导出 MZ3 格式")

    result = {"TI_file": mz3_path}
    logger.debug("TI_file: %s", mz3_path)
    send_progress(message_queue, task_id, "forward", 100, "已完成: 仿真完成", result)


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
    output_dir = os.path.join(params.dir_path, "TI_optimization", task_id)
    anisotropy_type = "vn" if params.anisotropy else "scalar"
    reset_task_output_dir(output_dir)

    # 获取 EEG 电极帽 CSV 文件路径（用于电极映射）
    net_electrode_file = find_montage_file(params.dir_path, params.montage)

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
    send_progress(message_queue, task_id, "inverse", 0, "任务开始")

    # 10% - init_optimization
    opt = init_optimization(
        subject_dir=params.dir_path,
        msh_file_path=params.msh_file_path,
        output_dir=output_dir,
        anisotropy_type=anisotropy_type,
        cond=cond_dict_to_list(params.cond),
        fname_tensor=params.DTI_file_path,
    )
    send_progress(message_queue, task_id, "inverse", 10, "已完成: 初始化优化结构")

    # 20% - setup_goal
    setup_goal(
        opt=opt,
        goal="focality",
        focality_threshold=[focality_threshold, NON_ROI_THRESHOLD],
        net_electrode_file=net_electrode_file,
    )
    send_progress(message_queue, task_id, "inverse", 20, "已完成: 配置目标函数")

    # 35% - setup_electrodes_and_roi
    setup_electrodes_and_roi(
        opt=opt,
        goal="focality",
        electrode_pair1_center=params.electrode_pair1_center,
        electrode_pair2_center=params.electrode_pair2_center,
        electrode_radius=params.electrode_radius,
        electrode_current1=params.electrode_current1,
        electrode_current2=params.electrode_current2,
        roi_center=roi_center,
        roi_radius=roi_radius,
        roi_center_space=roi_center_space,
        roi_mask_path=roi_mask_path,
        roi_mask_space=roi_mask_space,
        non_roi_center=None,
        non_roi_radius=None,
    )
    send_progress(message_queue, task_id, "inverse", 35, "已完成: 配置电极对和 ROI")

    # 85% - run_optimization
    run_optimization(opt=opt, n_workers=N_WORKERS)
    send_progress(message_queue, task_id, "inverse", 85, "已完成: 优化算法执行")

    # 90% - get_electrode_mapping
    electrode_A, electrode_B = get_electrode_mapping(output_dir=output_dir)
    send_progress(message_queue, task_id, "inverse", 90, "已完成: 获取电极映射结果")

    # 95% - export_mz3
    mz3_path = optimize_export_mz3(
        output_dir=output_dir,
        surface_type="central",
    )
    send_progress(message_queue, task_id, "inverse", 95, "已完成: 导出 MZ3 格式")

    result = {
        "TI_file": mz3_path,
        "electrode_A": electrode_A,
        "electrode_B": electrode_B,
    }
    logger.debug(
        "TI_file: %s, electrode_A: %s, electrode_B: %s",
        mz3_path,
        electrode_A,
        electrode_B,
    )
    send_progress(message_queue, task_id, "inverse", 100, "已完成: 优化完成", result)


def handle_ack_test_task(message_queue: Queue, task_id: str, params: Any) -> None:
    """处理 ack 时机验证任务。"""
    send_progress(message_queue, task_id, "ack_test", 0, "任务开始")
    logger.info(
        "worker_started: task_id=%s type=ack_test sleep_seconds=%.2f",
        task_id,
        params.sleep_seconds,
    )
    time.sleep(params.sleep_seconds)
    logger.info("worker_finished: task_id=%s type=ack_test", task_id)
    result = {"sleep_seconds": params.sleep_seconds}
    send_progress(
        message_queue, task_id, "ack_test", 100, "已完成: ack_test 完成", result
    )


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
