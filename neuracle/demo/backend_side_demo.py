"""
Backend 端 RabbitMQ 测试示例

持续运行，发送消息到 SimNIBS 并监听返回结果。
启动方式: python -m neuracle.demo.backend_side_demo
"""

import json
import logging
import threading
import time

from neuracle.logger import setup_logging
from neuracle.rabbitmq import RabbitMQListener, RabbitMQSender
from neuracle.rabbitmq.message_builder import (
    build_forward_message,
    build_inverse_message,
    build_model_message,
)
from neuracle.utils.env import get_rabbitmq_config, load_env

load_env()
logger = logging.getLogger(__name__)

# 配置
config = get_rabbitmq_config()


def result_callback(channel, method, properties, body):
    """接收 SimNIBS 返回的结果消息"""
    try:
        message = json.loads(body.decode("utf-8"))
        logger.info("=" * 60)
        logger.info("收到结果消息:")
        logger.info("  ID: %s", message.get("id"))
        logger.info("  Type: %s", message.get("type"))
        logger.info("  Progress: %s%%", message.get("progress_rate"))
        if message.get("message"):
            logger.info("  Message: %s", message.get("message"))
        if message.get("result"):
            logger.info(
                "  Result: %s",
                json.dumps(message.get("result"), indent=4, ensure_ascii=False),
            )
        logger.info("=" * 60)

        # 确认消息
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except json.JSONDecodeError as e:
        logger.error("JSON 解析失败: %s", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except Exception as e:
        logger.error("处理结果消息时发生错误: %s", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def listener_thread(queue_name):
    """Listener 线程函数，持续监听结果队列"""
    listener = RabbitMQListener(
        host=config["host"],
        port=config["port"],
        queue_name=queue_name,
        username=config["username"],
        password=config["password"],
        virtual_host=config["virtual_host"],
        heartbeat=config["heartbeat"],
        blocked_connection_timeout=config["blocked_connection_timeout"],
        socket_timeout=config["socket_timeout"],
        connection_attempts=config["connection_attempts"],
        retry_delay=config["retry_delay"],
    )

    logger.info("Result Listener 已启动...")

    try:
        listener.consume_forever(result_callback)
    except KeyboardInterrupt:
        logger.info("Result Listener 收到中断信号")
    finally:
        listener.close()
        logger.info("Result Listener 已停止")


def run_backend_side():
    """启动 Backend 端（交互模式）"""
    logger.info("启动 Backend 端...")
    logger.info("发送队列: %s", config["send_queue_name"])
    logger.info("监听队列: %s", config["listen_queue_name"])

    # 启动 listener 线程监听结果
    listener = threading.Thread(
        target=listener_thread,
        args=(config["listen_queue_name"],),
        daemon=True,
    )
    listener.start()
    time.sleep(0.5)  # 等待 listener 启动

    run_interactive_mode()

    logger.info("消息发送完成，继续监听结果（等待 SimNIBS 处理）...")
    logger.info("按 Ctrl+C 停止")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")


def run_interactive_mode():
    """交互模式：显示菜单供用户选择发送的消息"""
    test_messages = get_test_messages()
    error_messages = get_error_messages()

    while True:
        print("\n" + "=" * 50)
        print("请选择操作：")
        print("-" * 50)
        print("  1. 发送所有正常消息")
        print("  2. 发送所有错误消息")
        print("  3. 退出")
        print("-" * 50)

        choice = input("请输入选项 (1-3): ").strip()

        if choice == "1":
            for name, msg in test_messages:
                logger.info("发送消息: %s", name)
                send_one_message(msg)
                time.sleep(0.3)
            logger.info("所有正常消息发送完成")

        elif choice == "2":
            for name, msg in error_messages:
                logger.info("发送错误消息: %s", name)
                send_one_message(msg)
                time.sleep(0.3)
            logger.info("所有错误消息发送完成")

        elif choice == "3":
            print("退出交互模式")
            break

        else:
            print("无效的选项")


def send_one_message(message: dict) -> bool:
    """按当前发送策略，单条消息单次建连发送。"""
    sender = RabbitMQSender(
        host=config["host"],
        port=config["port"],
        queue_name=config["send_queue_name"],
        username=config["username"],
        password=config["password"],
        virtual_host=config["virtual_host"],
        heartbeat=config["heartbeat"],
        blocked_connection_timeout=config["blocked_connection_timeout"],
        socket_timeout=config["socket_timeout"],
        connection_attempts=config["connection_attempts"],
        retry_delay=config["retry_delay"],
    )
    try:
        if not sender.connect():
            logger.error("Sender 连接失败")
            return False
        return sender.send_message(message)
    finally:
        sender.close()


def get_test_messages():
    """获取所有正常测试消息"""
    messages = []

    # 1. 头模生成 - 必填参数（仅 T1）
    msg1 = build_model_message(
        id="test_model_001",
        dir_path="/tmp/test_model_001",
        T1_file_path="/tmp/test_t1.nii.gz",
    )
    messages.append(("model_必填参数", msg1))

    # 2. 头模生成 - 包含可选参数（T2 + DTI）
    msg2 = build_model_message(
        id="test_model_002",
        dir_path="/tmp/test_model_002",
        T1_file_path="/tmp/test_t1.nii.gz",
        T2_file_path="/tmp/test_t2.nii.gz",
        DTI_file_path="/tmp/test_dti.nii.gz",
    )
    messages.append(("model_可选参数", msg2))

    # 3. 正向仿真 - anisotropy=False
    msg3 = build_forward_message(
        id="test_forward_001",
        dir_path="/tmp/test_forward",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/test_montage.csv",
        electrode_A=["F5", "P5"],
        electrode_B=["F5", "P5"],
        current_A=[0.002, -0.002],
        current_B=[0.001, -0.001],
        cond={"White Matter": 0.126, "Gray Matter": 0.275},
        anisotropy=False,
    )
    messages.append(("forward_anisotropy_false", msg3))

    # 4. 正向仿真 - anisotropy=True with DTI
    msg4 = build_forward_message(
        id="test_forward_002",
        dir_path="/tmp/test_forward",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/test_montage.csv",
        electrode_A=["F5", "P5"],
        electrode_B=["P5", "F5"],
        current_A=[0.001, -0.001],
        current_B=[0.001, -0.001],
        cond={"White Matter": 0.126, "Gray Matter": 0.275, "CSF": 1.654},
        anisotropy=True,
        DTI_file_path="/tmp/test_dti.nii.gz",
    )
    messages.append(("forward_anisotropy_true", msg4))

    # 5. 正向仿真 - anisotropy=False with DTI
    msg5 = build_forward_message(
        id="test_forward_004",
        dir_path="/tmp/test_forward",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/test_montage.csv",
        electrode_A=["F5", "P5"],
        electrode_B=["F5", "P5"],
        current_A=[0.002, -0.002],
        current_B=[0.001, -0.001],
        cond={"White Matter": 0.126, "Gray Matter": 0.275},
        anisotropy=False,
        DTI_file_path="/tmp/test_dti.nii.gz",
    )
    messages.append(("forward_anisotropy_false_with_dti", msg5))

    # 6. 正向仿真 - 多电极配置
    msg6 = build_forward_message(
        id="test_forward_003",
        dir_path="/tmp/test_forward",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/test_montage.csv",
        electrode_A=["F3", "FC5", "FC1", "Cz"],
        electrode_B=["P3", "PO7", "PO3", "O1"],
        current_A=[0.001, 0.001, -0.001, -0.001],
        current_B=[-0.001, -0.001, 0.001, 0.001],
        cond={
            "White Matter": 0.126,
            "Gray Matter": 0.275,
            "CSF": 1.654,
            "Bone": 0.01,
            "Scalp": 0.465,
        },
        anisotropy=False,
    )
    messages.append(("forward_多电极", msg6))

    # 7. 逆向仿真 - Atlas ROI
    msg7 = build_inverse_message(
        id="test_inverse_001",
        dir_path="/tmp/test_inverse",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/test_montage.csv",
        current_A=[0.002, -0.002],
        current_B=[0.001, -0.001],
        roi_type="atlas",
        roi_param={
            "atlas_param": {"name": "AAL3", "area": "Precentral_L"},
            "mni_param": None,
        },
        target_threshold=0.5,
        cond={"White Matter": 0.126, "Gray Matter": 0.275},
        anisotropy=False,
    )
    messages.append(("inverse_atlas_roi", msg7))

    # 8. 逆向仿真 - MNI Position ROI
    msg8 = build_inverse_message(
        id="test_inverse_002",
        dir_path="/tmp/test_inverse",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/test_montage.csv",
        current_A=[0.001, -0.001],
        current_B=[0.001, -0.001],
        roi_type="mni_pos",
        roi_param={
            "atlas_param": None,
            "mni_param": {"center": [-38.5, -22.5, 58.3], "radius": 15.0},
        },
        target_threshold=0.7,
        cond={"White Matter": 0.126, "Gray Matter": 0.275, "CSF": 1.654},
        anisotropy=True,
        DTI_file_path="/tmp/test_dti.nii.gz",
    )
    messages.append(("inverse_mni_pos_roi", msg8))

    # 9. 逆向仿真 - 边界值测试（threshold=0）
    msg9 = build_inverse_message(
        id="test_inverse_003",
        dir_path="/tmp/test_inverse",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/test_montage.csv",
        current_A=[0.001, -0.001],
        current_B=[0.001, -0.001],
        roi_type="atlas",
        roi_param={
            "atlas_param": {"name": "AAL3", "area": "Postcentral_R"},
            "mni_param": None,
        },
        target_threshold=0.0,
        cond={"Gray Matter": 0.275},
        anisotropy=False,
    )
    messages.append(("inverse_threshold_边界值", msg9))

    return messages


def get_error_messages():
    """获取所有错误测试消息"""
    messages = []

    # 1. model - 缺少必填字段
    msg1 = {
        "id": "test_error_001",
        "type": "model",
        "params": {"T2_file_path": "/tmp/test.nii.gz"},
    }
    messages.append(("model_缺少必填字段", msg1))

    # 2. model - id 为空
    msg2 = {
        "id": "",
        "type": "model",
        "params": {"T1_file_path": "/tmp/test.nii.gz", "dir_path": "/tmp"},
    }
    messages.append(("model_id为空", msg2))

    # 3. forward - electrode_A 和 current_A 长度不一致
    msg3 = build_forward_message(
        id="test_error_002",
        dir_path="/tmp",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/montage.csv",
        electrode_A=["F5", "P5", "P3"],
        electrode_B=["F5", "P5"],
        current_A=[0.001, -0.001],
        current_B=[0.001, -0.001],
        cond={"Gray Matter": 0.275},
        anisotropy=False,
    )
    messages.append(("forward_电极电流长度不一致", msg3))

    # 4. inverse - roi_type 非法值
    msg4 = build_inverse_message(
        id="test_error_003",
        dir_path="/tmp",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/montage.csv",
        current_A=[0.001, -0.001],
        current_B=[0.001, -0.001],
        roi_type="invalid_type",
        roi_param={
            "atlas_param": None,
            "mni_param": {"center": [-38.5, -22.5, 58.3], "radius": 15.0},
        },
        target_threshold=0.5,
        cond={"Gray Matter": 0.275},
        anisotropy=False,
    )
    messages.append(("inverse_roi_type非法", msg4))

    # 5. inverse - target_threshold 为负数
    msg5 = build_inverse_message(
        id="test_error_004",
        dir_path="/tmp",
        msh_file_path="/tmp/test.msh",
        montage="/tmp/montage.csv",
        current_A=[0.001, -0.001],
        current_B=[0.001, -0.001],
        roi_type="atlas",
        roi_param={
            "atlas_param": {"name": "AAL3", "area": "Precentral_L"},
            "mni_param": None,
        },
        target_threshold=-0.5,
        cond={"Gray Matter": 0.275},
        anisotropy=False,
    )
    messages.append(("inverse_threshold负数", msg5))

    # 6. 未知 type
    msg6 = {"id": "test", "type": "unknown", "params": {}}
    messages.append(("未知type", msg6))

    return messages


if __name__ == "__main__":
    load_env()
    setup_logging()
    run_backend_side()
