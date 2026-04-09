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

    def __init__(
        self,
        host: str,
        port: int,
        queue_name: str,
        message_queue: Queue,
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/",
        heartbeat: int = 60,
        blocked_connection_timeout: int = 300,
        socket_timeout: int = 10,
        connection_attempts: int = 5,
        retry_delay: int = 5,
    ):
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
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.heartbeat = heartbeat
        self.blocked_connection_timeout = blocked_connection_timeout
        self.socket_timeout = socket_timeout
        self.connection_attempts = connection_attempts
        self.retry_delay = retry_delay
        self.running = True
        self.sender: RabbitMQSender | None = None

    def run(self) -> None:
        """线程运行逻辑"""
        logger.info("发送线程启动")
        # 持续从队列获取消息并发送
        while self.running:
            try:
                # 空闲时不持有 RabbitMQ 连接，因此这里可以安全阻塞等待消息
                message = self.message_queue.get()
                if not self.running:
                    self.message_queue.task_done()
                    break

                self.sender = RabbitMQSender(
                    host=self.host,
                    port=self.port,
                    queue_name=self.queue_name,
                    username=self.username,
                    password=self.password,
                    virtual_host=self.virtual_host,
                    heartbeat=self.heartbeat,
                    blocked_connection_timeout=self.blocked_connection_timeout,
                    socket_timeout=self.socket_timeout,
                    connection_attempts=self.connection_attempts,
                    retry_delay=self.retry_delay,
                )

                if not self.sender.connect():
                    logger.error("发送器连接失败，当前消息发送失败")
                else:
                    self.sender.send_message(message)
            except Exception as e:
                if self.running:
                    logger.error("发送线程处理消息时发生错误: %s", e)
            finally:
                if self.sender:
                    self.sender.close()
                    self.sender = None
                if self.running:
                    self.message_queue.task_done()
        # 清理资源
        if self.sender:
            self.sender.close()
        logger.info("发送线程退出")

    def stop(self) -> None:
        """停止发送线程"""
        self.running = False
        self.message_queue.put(None)
