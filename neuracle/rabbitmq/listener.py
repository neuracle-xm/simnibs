"""
RabbitMQ 消息监听器模块

用于监听 RabbitMQ 队列，接收来自后端服务器的消息。
"""

import logging
import time
from typing import Any, Callable

from pika import BlockingConnection, ConnectionParameters, PlainCredentials
from pika.exceptions import AMQPConnectionError, ConnectionWrongStateError

logger = logging.getLogger(__name__)


class RabbitMQListener:
    """RabbitMQ 监听器类"""

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
        prefetch_count: int = 1,
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
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.heartbeat = heartbeat
        self.blocked_connection_timeout = blocked_connection_timeout
        self.socket_timeout = socket_timeout
        self.connection_attempts = connection_attempts
        self.retry_delay = retry_delay
        self.prefetch_count = prefetch_count
        self.connection: BlockingConnection | None = None
        self.channel = None
        self._stopped = False

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
            # 声明队列（持久化）
            self.channel.queue_declare(
                queue=self.queue_name,
                durable=True,
                arguments={"x-queue-type": "quorum", "x-consumer-timeout": 86400000},
            )
            logger.info(
                "成功连接到 RabbitMQ 服务器: host=%s port=%s vhost=%s user=%s queue=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
            )
            return True
        except AMQPConnectionError as e:
            logger.error(
                "连接 RabbitMQ 失败: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s error=%r",
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
                "连接时发生未知错误: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
                type(e).__name__,
            )
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
        self.channel.basic_qos(prefetch_count=self.prefetch_count)  # type: ignore
        self.channel.basic_consume(
            queue=self.queue_name, on_message_callback=callback, auto_ack=False
        )  # type: ignore
        self.channel.start_consuming()  # type: ignore

    def consume_forever(self, callback: Callable[[Any, Any, Any, bytes], None]) -> None:
        """持续消费消息，断线后自动重连。"""
        self._stopped = False
        while not self._stopped:
            if not self.connect():
                if self._stopped:
                    break
                logger.warning(
                    "监听器连接失败，%s 秒后重试: queue=%s",
                    self.retry_delay,
                    self.queue_name,
                )
                time.sleep(self.retry_delay)
                continue
            try:
                self.start_consume(callback)
            except KeyboardInterrupt:
                self._stopped = True
                raise
            except Exception as e:
                if not self._stopped:
                    logger.exception(
                        "监听器消费中断，准备重连: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s",
                        self.host,
                        self.port,
                        self.virtual_host,
                        self.username,
                        self.queue_name,
                        type(e).__name__,
                    )
            finally:
                self.close()
            if not self._stopped:
                time.sleep(self.retry_delay)

    def stop_consume(self) -> None:
        """停止消费消息"""
        self._stopped = True
        if self.channel:
            self.channel.stop_consuming()

    def close(self) -> None:
        """关闭连接"""
        if self.connection and not self.connection.is_closed:
            try:
                self.connection.close()
            except ConnectionWrongStateError:
                logger.warning("RabbitMQ 连接已处于关闭流程，忽略重复关闭")
            except Exception as e:
                logger.warning("关闭 RabbitMQ 监听连接时发生错误: %s", e)
        self.channel = None
