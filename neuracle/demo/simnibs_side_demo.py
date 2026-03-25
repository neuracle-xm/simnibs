"""
SimNIBS 端 RabbitMQ 测试示例

持续运行，接收来自 Backend 的消息，验证参数，发送进度和结果消息。
启动方式: python -m neuracle.demo.simnibs_side_demo
"""

import json
import logging
from functools import partial
from queue import Queue

from neuracle.env import get_rabbitmq_config, load_env
from neuracle.logger import setup_logging
from neuracle.rabbitmq import RabbitMQListener, SenderThread
from neuracle.rabbitmq.message_builder import build_progress_message
from neuracle.rabbitmq.validator import ValidationError, validate_message

logger = logging.getLogger(__name__)


def handle_message(message_queue: Queue, channel, method, properties, body):
    """
    处理接收到的消息，验证并发送响应

    Args:
        message_queue: 消息队列，用于发送响应消息
        channel: pika.Channel 通道对象
        method: pika.spec.Basic.Deliver 消息传递信息
        properties: pika.spec.BasicProperties 消息属性
        body: bytes 消息内容
    """
    try:
        message = json.loads(body.decode("utf-8"))
        msg_id = message.get("id")
        msg_type = message.get("type")
        params = message.get("params", {})

        logger.info("=" * 60)
        logger.info("收到消息:")
        logger.info("  ID: %s", msg_id)
        logger.info("  Type: %s", msg_type)
        logger.info(
            "  Params: %s",
            json.dumps(params, indent=4, ensure_ascii=False),
        )
        logger.info("=" * 60)

        # 验证消息
        try:
            validate_message(message)
            logger.info("消息验证通过")

            # 发送进行中进度
            progress_msg = build_progress_message(
                id=msg_id,
                msg_type=msg_type,
                progress_rate=50,
                message="Processing...",
                result=None,
            )
            message_queue.put(progress_msg)
            logger.info("已发送进度消息: progress_rate=50")

            # 模拟处理完成，发送完成消息
            if msg_type == "model":
                result = {"msh_file_path": f"/data/{msg_id}/{msg_id}.msh"}
            elif msg_type == "forward":
                result = {
                    "T1_mni": f"/data/{msg_id}/T1_mni.nii.gz",
                    "TI_file": f"/data/{msg_id}/forward_result.mz3",
                }
            else:  # inverse
                result = {
                    "T1_mni": f"/data/{msg_id}/T1_mni.nii.gz",
                    "TI_file": f"/data/{msg_id}/inverse_result.mz3",
                    "electrode_A": ["F3", "FC5"],
                    "electrode_B": ["P3", "PO7"],
                }

            complete_msg = build_progress_message(
                id=msg_id,
                msg_type=msg_type,
                progress_rate=100,
                message=None,
                result=result,
            )
            message_queue.put(complete_msg)
            logger.info("已发送完成消息: progress_rate=100")

        except ValidationError as e:
            logger.warning("消息验证失败: %s", e)

            # 发送错误消息
            error_msg = build_progress_message(
                id=msg_id,
                msg_type=msg_type,
                progress_rate=0,
                message=str(e),
                result=None,
            )
            message_queue.put(error_msg)
            logger.info("已发送错误消息")

        # 确认消息
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except json.JSONDecodeError as e:
        logger.error("JSON 解析失败: %s", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except Exception as e:
        logger.error("处理消息时发生错误: %s", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def run_simnibs_side(config):
    """启动 SimNIBS 端，持续监听消息

    Args:
        config: RabbitMQ 配置字典
    """
    logger.info("启动 SimNIBS 端...")
    logger.info("监听队列: %s", config["send_queue_name"])
    logger.info("发送队列: %s", config["listen_queue_name"])

    # 创建消息队列
    message_queue = Queue()

    # 创建 sender 线程
    sender_thread = SenderThread(
        host=config["host"],
        port=config["port"],
        queue_name=config["listen_queue_name"],
        message_queue=message_queue,
    )
    sender_thread.start()

    # 创建 listener
    listener = RabbitMQListener(
        host=config["host"],
        port=config["port"],
        queue_name=config["send_queue_name"],
    )

    if not listener.connect():
        logger.error("Listener 连接失败")
        sender_thread.stop()
        return

    logger.info("SimNIBS 端已启动，持续监听消息（Ctrl+C 退出）...")

    try:
        listener.start_consume(partial(handle_message, message_queue))
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    finally:
        sender_thread.stop()
        sender_thread.join(timeout=5)
        listener.close()
        logger.info("SimNIBS 端已停止")


if __name__ == "__main__":
    load_env()
    setup_logging()
    config = get_rabbitmq_config()
    run_simnibs_side(config)
