"""
RabbitMQ 发送线程模块

提供专门的发送线程，用于从队列中获取消息并发送。
"""

import logging
import threading
from queue import Queue

from neuracle.rabbitmq.sender import RabbitMQSender

logger = logging.getLogger(__name__)


class SenderThread(threading.Thread):
    """专门用于发送消息的线程"""

    def __init__(self, host: str, port: int, queue_name: str, message_queue: Queue):
        """
        初始化发送线程

        Args:
            host: RabbitMQ 服务器地址
            port: RabbitMQ 服务器端口
            queue_name: 发送队列名称
            message_queue: Python Queue，存放待发送的消息
        """
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.queue_name = queue_name
        self.message_queue = message_queue
        self.running = True
        self.sender: RabbitMQSender | None = None

    def run(self) -> None:
        """线程运行逻辑"""
        logger.info("发送线程启动")
        # 创建发送器
        self.sender = RabbitMQSender(
            host=self.host,
            port=self.port,
            queue_name=self.queue_name,
        )
        # 连接 RabbitMQ
        if not self.sender.connect():
            logger.error("发送器连接失败，发送线程退出")
            return
        # 持续从队列获取消息并发送
        while self.running:
            try:
                # 从队列获取消息（阻塞等待）
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
