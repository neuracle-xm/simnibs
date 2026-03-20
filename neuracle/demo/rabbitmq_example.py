"""
RabbitMQ 监听器使用示例

演示如何使用 RabbitMQListener 监听来自后端服务器的消息，
并通过专门的发送线程发送响应消息。
配置从 .env 文件读取。
"""

import json
import logging
from queue import Queue
import threading
from neuracle.logger import setup_logging
from neuracle.rabbitmq import RabbitMQListener, RabbitMQSender
from neuracle.env import load_env, get_rabbitmq_config
from functools import partial

# 加载环境变量
load_env()

# 获取 logger
logger = logging.getLogger('neuracle.demo')


class SenderThread(threading.Thread):
    """专门用于发送消息的线程"""

    def __init__(self, config: dict, message_queue:Queue):
        """
        初始化发送线程

        Args:
            config: RabbitMQ 配置
            message_queue: 消息队列
        """
        super().__init__(daemon=True)
        self.config = config
        self.message_queue = message_queue
        self.running = True
        self.sender: RabbitMQSender | None = None

    def run(self) -> None:
        """线程运行逻辑"""
        logger.info("发送线程启动")
        # 创建发送器
        self.sender = RabbitMQSender(
            host=self.config['host'],
            port=self.config['port'],
            queue_name=self.config['send_queue_name']
        )
        # 连接 RabbitMQ
        if not self.sender.connect():
            logger.error("发送器连接失败，发送线程退出")
            return
        # 持续从队列获取消息并发送
        while self.running:
            try:
                # 从队列获取消息
                message = self.message_queue.get()
                # 发送消息
                if self.sender:
                    self.sender.send_message(message)
                self.message_queue.task_done()
            except Exception as e:
                if self.running:
                    logger.error("发送线程处理消息时发生错误: %s", e)
        # 清理资源
        if self.sender:
            self.sender.close()
        logger.info("发送线程退出")

    def stop(self) -> None:
        """停止发送线程"""
        self.running = False


def message_callback(channel, method, properties, body, message_queue:Queue):
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
        message = json.loads(body.decode('utf-8'))
        logger.info("收到消息: %s", message)
        # 构造响应消息
        response = {
            'type': 'response',
            'original_type': message.get('type', 'unknown'),
            'status': 'processed',
            'message': '已处理消息'
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
    logger.info("RabbitMQ 配置: %s", config)
    message_queue = Queue()
    # 创建sender
    sender_thread = SenderThread(config, message_queue)
    sender_thread.start()
    # 创建listener
    listener = RabbitMQListener(
        host=config['host'],
        port=config['port'],
        queue_name=config['listen_queue_name']
    )
    # 使用 partial 绑定自定义参数
    callback_with_args = partial(message_callback, message_queue=message_queue)
    # 连接并开始消费
    if listener.connect():
        logger.info("开始监听消息...")
        try:
            listener.start_consume(callback_with_args)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
    else:
        logger.error("连接失败，请检查 RabbitMQ 服务器是否运行")
    # 停止发送线程
    sender_thread.stop()
    sender_thread.join(timeout=5)
    logger.info("程序退出")


if __name__ == "__main__":
    main()
