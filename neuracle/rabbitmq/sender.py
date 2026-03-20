"""
RabbitMQ 消息发送模块

用于向 RabbitMQ 队列发送消息。
"""

import json
from typing import Any

from pika import BlockingConnection, ConnectionParameters, BasicProperties, DeliveryMode
from pika.exceptions import AMQPConnectionError

import logging

logger = logging.getLogger(__name__)


class RabbitMQSender:
    """RabbitMQ 发送器类"""

    def __init__(self,
                 host: str = 'localhost',
                 port: int = 5672,
                 queue_name: str = ''):
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
        self.connection: BlockingConnection | None = None
        self.channel = None

    def connect(self) -> bool:
        """
        连接到 RabbitMQ 服务器

        Returns:
            bool: 连接是否成功
        """
        try:
            # 连接参数
            parameters = ConnectionParameters(host=self.host, port=self.port)
            # 创建连接
            self.connection = BlockingConnection(parameters)
            self.channel = self.connection.channel()
            # 如果指定了队列名，声明队列
            if self.queue_name:
                self.channel.queue_declare(
                    queue=self.queue_name,
                    durable=True,
                    arguments={'x-queue-type': 'quorum'}
                )
            logger.info("发送器成功连接到 RabbitMQ 服务器: %s:%s", self.host, self.port)
            return True
        except AMQPConnectionError as e:
            logger.error("发送器连接 RabbitMQ 失败: %s", e)
            return False
        except Exception as e:
            logger.error("发送器连接时发生未知错误: %s", e)
            return False

    def send_message(self,
                     message: Any) -> bool:
        """
        发送消息

        原理：
            将消息序列化为 JSON 格式，通过 basic_publish 方法发送到指定队列或交换机。

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
            body = json.dumps(message, ensure_ascii=False).encode('utf-8')
            # 设置消息属性
            properties = BasicProperties(
                delivery_mode=DeliveryMode.Persistent,  # 持久化
            )
            # 发送消息
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=body,
                properties=properties
            )
            logger.info("消息已发送: %s", message)
            return True
        except Exception as e:
            logger.error("发送消息失败: %s", e)
            return False

    def close(self) -> None:
        """关闭连接"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("发送器连接已关闭")
        else:
            logger.warning("发送器还未连接时就调用close")
