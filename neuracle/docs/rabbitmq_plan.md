# RabbitMQ 功能规划

## 一、现状分析

### 1.1 已实现功能

| 模块     | 文件                       | 功能                                                             |
| -------- | -------------------------- | ---------------------------------------------------------------- |
| 监听器   | `rabbitmq/listener.py`     | `RabbitMQListener` 类，基于 pika BlockingConnection 实现队列监听 |
| 发送器   | `rabbitmq/sender.py`       | `RabbitMQSender` 类，基于 pika BlockingConnection 实现消息发送   |
| 配置管理 | `env.py`                   | 从 `.env` 文件读取 RabbitMQ 配置（host、port、队列名）           |
| 示例     | `demo/rabbitmq_example.py` | 完整的收发示例，包含独立发送线程避免阻塞                         |

### 1.2 当前架构

```
后端服务 <---> RabbitMQ Server <---> RabbitMQListener (消费)
                                         |
                                   Queue (待处理消息)

RabbitMQSender (发送) <---> Queue <---> 后端服务
```

### 1.3 现有代码特点

- **连接方式**: 使用 `BlockingConnection`，适合同步场景
- **队列类型**: 声明为 `quorum` 类型（高可靠性）
- **消息持久化**: 开启 `DeliveryMode.Persistent`
- **确认机制**: 手动 `basic_ack` / `basic_nack`
- **配置分离**: 通过 `.env` 文件管理敏感配置

---