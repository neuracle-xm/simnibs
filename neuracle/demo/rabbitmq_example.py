"""
RabbitMQ 监听器使用示例

演示如何使用 RabbitMQListener 监听来自后端服务器的消息，
并通过专门的发送线程发送响应消息。
配置从 .env 文件读取。
"""

import json
import logging
from functools import partial
from queue import Queue

from neuracle.logger import setup_logging
from neuracle.rabbitmq import RabbitMQListener, SenderThread
from neuracle.utils.env import get_rabbitmq_config, load_env

# 加载环境变量
load_env()

# 获取 logger
logger = logging.getLogger(__name__)


def mask_rabbitmq_config(config: dict) -> dict:
    """隐藏敏感配置，避免密码写入日志。"""
    masked = dict(config)
    if masked.get("password"):
        masked["password"] = "***"
    return masked


def message_callback(channel, method, properties, body, message_queue: Queue):
    """
    消息处理回调函数

    原理：
        收到消息后解析并放入发送队列，由专门的发送线程处理。
        这样可以避免阻塞主线程的监听循环，同时复用发送连接。

    Args:
        channel: pika.Channel 通道对象
        method: pika.spec.Basic.Deliver 消息传递信息
        properties: pika.spec.BasicProperties 消息属性
        body: bytes 消息内容
        message_queue: 用于把结果作为消息返回给发送消息的线程
    """
    try:
        # 解析消息
        message = json.loads(body.decode("utf-8"))
        logger.info("收到消息: %s", message)
        # 构造响应消息
        response = {
            "type": "response",
            "original_type": message.get("type", "unknown"),
            "status": "processed",
            "message": "已处理消息",
        }
        message_queue.put(response)
        logger.info("响应已加入发送队列")
        # 立即确认消息
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except json.JSONDecodeError as e:
        logger.error("无法解析 JSON 消息: %s", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except Exception as e:
        logger.error("处理消息时发生错误: %s", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    """主函数"""
    setup_logging()
    # 从 .env 读取配置
    config = get_rabbitmq_config()
    logger.info("RabbitMQ 配置: %s", mask_rabbitmq_config(config))
    message_queue = Queue()
    # 创建 sender 线程
    sender_thread = SenderThread(
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
    sender_thread.start()
    # 创建 listener
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
    # 使用 partial 绑定自定义参数
    callback_with_args = partial(message_callback, message_queue=message_queue)
    # 连接并开始消费
    logger.info("开始监听消息...")
    try:
        listener.consume_forever(callback_with_args)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    # 停止发送线程
    sender_thread.stop()
    sender_thread.join(timeout=5)
    logger.info("程序退出")


if __name__ == "__main__":
    main()
