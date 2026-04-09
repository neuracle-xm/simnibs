"""
RabbitMQ 消息发送模块

用于向 RabbitMQ 队列发送消息。
"""

import json
import logging
from typing import Any

from pika import (
    BasicProperties,
    BlockingConnection,
    ConnectionParameters,
    DeliveryMode,
    PlainCredentials,
)
from pika.exceptions import AMQPConnectionError

logger = logging.getLogger(__name__)


class RabbitMQSender:
    """RabbitMQ 发送器类"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        queue_name: str = "",
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
        初始化 RabbitMQ 发送器

        Args:
            host: RabbitMQ 服务器地址
            port: RabbitMQ 服务器端口
            queue_name: 队列名称（直接发送到队列）
        """
        self.host = host
        self.port = port
        self.queue_name = queue_name
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.heartbeat = heartbeat
        self.blocked_connection_timeout = blocked_connection_timeout
        self.socket_timeout = socket_timeout
        self.connection_attempts = connection_attempts
        self.retry_delay = retry_delay
        self.connection: BlockingConnection | None = None
        self.channel = None

    def _build_connection_parameters(self) -> ConnectionParameters:
        """构造 RabbitMQ 连接参数。"""
        credentials = PlainCredentials(self.username, self.password)
        return ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=credentials,
            heartbeat=self.heartbeat,
            blocked_connection_timeout=self.blocked_connection_timeout,
            socket_timeout=self.socket_timeout,
            connection_attempts=self.connection_attempts,
            retry_delay=self.retry_delay,
        )

    def connect(self) -> bool:
        """
        连接到 RabbitMQ 服务器

        Returns:
            bool: 连接是否成功
        """
        try:
            parameters = self._build_connection_parameters()
            # 创建连接
            self.connection = BlockingConnection(parameters)
            self.channel = self.connection.channel()
            # 如果指定了队列名，声明队列
            if self.queue_name:
                self.channel.queue_declare(
                    queue=self.queue_name,
                    durable=True,
                    arguments={
                        "x-queue-type": "quorum",
                        "x-consumer-timeout": 86400000,
                    },
                )
            logger.info(
                "发送器成功连接到 RabbitMQ 服务器: host=%s port=%s vhost=%s user=%s queue=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
            )
            return True
        except AMQPConnectionError as e:
            logger.error(
                "发送器连接 RabbitMQ 失败: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s error=%r",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
                type(e).__name__,
                e,
            )
            return False
        except Exception as e:
            logger.exception(
                "发送器连接时发生未知错误: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
                type(e).__name__,
            )
            return False

    def send_message(self, message: Any) -> bool:
        """
        发送消息

        原理：
            将消息序列化为 JSON 格式，通过 basic_publish 方法发送到指定队列或交换机。
            如果通道已关闭，自动重连后重试发送。

        Args:
            message: 消息内容（可序列化为 JSON 的对象）

        Returns:
            bool: 发送是否成功
        """
        if not self.channel:
            logger.error("通道未连接，无法发送消息")
            return False
        try:
            # 序列化消息
            body = json.dumps(message, ensure_ascii=False).encode("utf-8")
            # 设置消息属性
            properties = BasicProperties(
                delivery_mode=DeliveryMode.Persistent,  # 持久化
            )
            # 发送消息
            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=body,
                properties=properties,
            )
            logger.info("消息已发送: %s", message)
            return True
        except Exception as e:
            logger.warning("发送消息失败: %s，尝试重连...", e)
            # 尝试重连
            if self._reconnect():
                try:
                    self.channel.basic_publish(
                        exchange="",
                        routing_key=self.queue_name,
                        body=body,
                        properties=properties,
                    )
                    logger.info("重连后消息发送成功: %s", message)
                    return True
                except Exception as e2:
                    logger.error("重连后发送消息仍然失败: %s", e2)
                    return False
            return False

    def _reconnect(self) -> bool:
        """重新建立连接"""
        try:
            logger.info(
                "尝试重新连接 RabbitMQ: host=%s port=%s vhost=%s user=%s queue=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
            )
            # 关闭旧连接
            if self.connection and self.connection.is_open:
                try:
                    self.connection.close()
                except Exception:
                    pass
            # 重新连接
            parameters = self._build_connection_parameters()
            self.connection = BlockingConnection(parameters)
            self.channel = self.connection.channel()
            # 重新声明队列
            if self.queue_name:
                self.channel.queue_declare(
                    queue=self.queue_name,
                    durable=True,
                    arguments={"x-queue-type": "quorum"},
                )
            logger.info(
                "重连成功: host=%s port=%s vhost=%s user=%s queue=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
            )
            return True
        except Exception as e:
            logger.exception(
                "重连失败: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
                type(e).__name__,
            )
            return False

    def close(self) -> None:
        """关闭连接"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("发送器连接已关闭")
        else:
            logger.warning("发送器还未连接时就调用close")
