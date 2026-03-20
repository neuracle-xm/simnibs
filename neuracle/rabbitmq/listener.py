"""
RabbitMQ 消息监听器模块

用于监听 RabbitMQ 队列，接收来自后端服务器的消息。
"""

from typing import Callable, Any

from pika import BlockingConnection, ConnectionParameters
from pika.exceptions import AMQPConnectionError

import logging

logger = logging.getLogger(__name__)


class RabbitMQListener:
    """RabbitMQ 监听器类"""

    def __init__(self,
                 host: str = 'localhost',
                 port: int = 5672,
                 queue_name: str = ''
                 ):
        """
        初始化 RabbitMQ 监听器

        Args:
            host: RabbitMQ 服务器地址
            port: RabbitMQ 服务器端口
            queue_name: 队列名称
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
            # 声明队列（持久化）
            self.channel.queue_declare(
                queue=self.queue_name,
                durable=True,
                arguments={'x-queue-type': 'quorum'}
            )
            logger.info("成功连接到 RabbitMQ 服务器: %s:%s", self.host, self.port)
            return True
        except AMQPConnectionError as e:
            logger.error("连接 RabbitMQ 失败: %s", e)
            return False
        except Exception as e:
            logger.error("连接时发生未知错误: %s", e)
            return False

    def start_consume(self, callback: Callable[[Any, Any, Any, bytes], None]) -> None:
        """
        开始消费消息

        原理：
            使用 pika 的 basic_consume 方法注册回调函数，然后调用 start_consuming
            进入阻塞循环，持续接收并处理消息。

        Args:
            callback: 消息回调函数，签名为 callback(channel, method, properties, body)
                - channel: pika.Channel 通道对象
                - method: pika.spec.Basic.Deliver 消息传递信息
                - properties: pika.spec.BasicProperties 消息属性
                - body: bytes 消息内容
        """
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=callback, auto_ack=False) # type: ignore
        self.channel.start_consuming() # type: ignore
