# RabbitMQ 功能规划

## 一、当前现状

### 1.1 当前项目中的实际使用方式

当前 RabbitMQ 相关实现主要由以下模块组成：

- 监听器：[neuracle/rabbitmq/listener.py](/C:/Users/50609/Documents/simnibs/neuracle/rabbitmq/listener.py)
- 发送线程：[neuracle/rabbitmq/sender_thread.py](/C:/Users/50609/Documents/simnibs/neuracle/rabbitmq/sender_thread.py)
- 主任务入口：[neuracle/main.py](/C:/Users/50609/Documents/simnibs/neuracle/main.py)

当前消费模型是：

1. `RabbitMQListener` 使用 `pika.BlockingConnection`
2. `channel.basic_consume(..., auto_ack=False)` 手动确认消息
3. `main.py` 的 `handle_message()` 在消费线程中解析消息
4. 校验通过后创建后台工作线程 `threading.Thread`
5. **当前实现会在 `handle_message()` 的 `finally` 中立即 `basic_ack`**
6. 真正耗时任务随后才在工作线程中执行

### 1.2 当前问题

当前实现的核心问题是：

- RabbitMQ 收到 `ack` 时，任务其实还没有执行完成
- 如果任务线程在中途崩溃，RabbitMQ 不会重新投递该消息
- 当前监听器没有显式设置 `prefetch_count`
- 在“任务尚未完成但消息已 ack”的前提下，Broker 可能继续把下一条消息推过来

也就是说，当前行为实际上是：

- `ack on dispatch`

而不是：

- `ack on completion`

## 二、目标调整

根据当前项目的需求，目标行为调整为：

1. **正常任务完成后再 ack**
2. **参数校验失败、消息解析失败、任务分发失败等异常分支仍然要 ack**
3. **不阻塞 RabbitMQ 消费线程**
4. **设置 `prefetch_count=1`，确保上一条消息未完成确认前，Broker 不再投递下一条消息**
5. **继续保留独立发送线程，避免进度回传阻塞监听线程**

## 三、推荐的消息确认模型

### 3.1 整体原则

应将确认语义改为：

- 主消费线程：只负责“接收消息、轻量校验、启动工作线程”
- 工作线程：负责执行实际任务
- 消费连接线程：负责真正调用 `basic_ack`

这里需要特别注意：

- `pika.BlockingConnection` / `channel` 不是随便在哪个线程都可以直接安全调用的
- 如果任务在线程池或后台线程里跑完后要确认消息，**不要直接在工作线程里调用 `channel.basic_ack(...)`**
- 更稳妥的做法是通过 `connection.add_callback_threadsafe(...)` 把 ack 操作投递回 RabbitMQ 消费连接所在的线程执行

### 3.2 推荐流程

#### A. 正常路径

1. `handle_message()` 收到消息
2. 在消费线程中完成：
   - `json.loads(body)`
   - `task_id` / `msg_type` 提取
   - 参数校验
   - 参数对象构造
3. 如果上述步骤都成功：
   - 创建工作线程
   - 将 `channel` / `delivery_tag` / `task_id` / `msg_type` 等必要上下文交给工作线程
4. `handle_message()` 立即返回，但**此时不 ack**
5. 工作线程执行实际任务
6. 任务成功完成后：
   - 通过 `connection.add_callback_threadsafe(...)`
   - 在消费线程里执行 `channel.basic_ack(delivery_tag=...)`

#### B. 异常路径

以下情况应视为“消息已经处理完毕，但处理结果是失败”，因此仍然 ack：

1. 消息 JSON 解析失败
2. `type` 非法
3. 参数校验失败
4. 参数对象构造失败
5. 工作线程启动失败
6. 工作线程内部任务执行抛异常并已发出失败进度

也就是说，当前项目的策略不是“异常就重新入队”，而是：

- 失败结果通过进度消息返回后端
- RabbitMQ 侧仍 ack，避免同一坏消息无限重试

这和当前 `send_progress(..., progress_rate=0, message=str(e))` 的用法是一致的。

## 四、为什么不能阻塞消费线程

当前监听器基于 `BlockingConnection.start_consuming()`。

如果把真正的长任务直接放在消费回调线程里执行，会带来几个问题：

1. `BlockingConnection` 所在线程长期卡在用户任务中
2. 心跳处理不及时，可能被 Broker 认为连接失活
3. 长任务期间无法及时处理连接关闭、异常回调等事件
4. 监听线程不能及时响应停止或重连逻辑

因此当前项目仍然应该保留：

- **消费线程只做轻量操作**
- **耗时任务放到后台线程执行**

这也是“完成后 ack”必须配合“线程安全回到消费线程做 ack”的原因。

## 五、prefetch_count 的要求

### 5.1 目标

需要在监听器中显式设置：

```python
channel.basic_qos(prefetch_count=1)
```

目标语义是：

- 当前这条消息没有被 ack 之前
- RabbitMQ 不再向该 consumer 投递下一条消息

### 5.2 这样做对当前项目的意义

在当前项目里，一个任务通常是：

- CHARM 建模
- TI 正向仿真
- TI 逆向优化

这些都属于长任务，且资源消耗较大。设置 `prefetch_count=1` 的好处是：

1. 保证单实例服务同一时刻只占用一条未确认消息
2. 避免在前一个任务还没完成时又收到下一个任务
3. 和“完成后 ack”的设计天然匹配
4. 降低内存、CPU、网格文件 IO 的并发压力

## 六、基于当前代码的改动建议

### 6.1 `handle_message()` 的改法

当前 `handle_message()` 的问题在于：

- `finally` 中无条件 `basic_ack(...)`

建议改成：

1. **仅在同步异常分支里 ack**
2. 正常派发到工作线程后，不在 `handle_message()` 中 ack
3. 把 ack 责任转移到工作线程完成后的回调逻辑

具体语义应为：

- 成功派发任务：暂不 ack
- 校验失败 / 解析失败 / 分发失败：立即 ack

### 6.2 `execute_task()` 的改法

当前 `execute_task()` 只负责：

- 运行任务
- 发送成功或失败进度

建议后续职责调整为：

1. 运行任务
2. 发送成功或失败进度
3. 无论成功还是失败，都在末尾触发 ack
4. ack 通过线程安全方式回到消费连接线程执行

### 6.3 工作线程如何把 ack 扔回消费线程

这里需要进一步明确：

- **不需要额外再建一个 Python `Queue` 专门传 ack**
- 当前项目只需要使用 `pika.BlockingConnection` 自带的 `add_callback_threadsafe(...)`

推荐做法如下。

#### A. 消费线程侧需要提前保存的信息

当 `handle_message()` 收到一条消息并准备派发到工作线程时，需要把下面这些信息一并交给工作线程：

1. `delivery_tag`
   - 来自 `method.delivery_tag`
   - 这是 RabbitMQ 最终执行 `basic_ack(...)` 时真正要确认的标识
2. `channel`
   - 用于最后执行 `basic_ack(...)`
3. `connection`
   - 推荐直接从 `channel.connection` 取得
   - 用于调用 `add_callback_threadsafe(...)`

注意：

- `task_id` 只是业务层标识，不能替代 `delivery_tag`
- RabbitMQ 的 ack 是针对当前 channel 下的 `delivery_tag`，不是针对 `task_id`

#### B. 工作线程侧的职责

工作线程结束时，不直接调用：

```python
channel.basic_ack(delivery_tag=delivery_tag)
```

而是只做两件事：

1. 定义一个很小的 `ack_callback`
2. 用 `connection.add_callback_threadsafe(ack_callback)` 把它投递回消费线程

逻辑上应理解为：

- 工作线程并不“自己 ack”
- 工作线程只是发起一个“请消费线程帮我 ack 这条消息”的请求

#### C. `ack_callback` 的职责

这个回调应该尽量保持很小，只做 RabbitMQ 确认本身。

它的典型职责是：

1. 检查 `channel` 是否仍然可用
2. 调用 `channel.basic_ack(delivery_tag=delivery_tag)`
3. 记录一条日志，说明哪条消息已经完成确认

也就是说，`ack_callback` 最好不要再做：

- 业务计算
- 文件操作
- 发送进度消息
- 复杂异常恢复

它应该只是一个“回到消费线程后执行 ack 的薄包装”。

#### D. 整体线程切换过程

完整过程可以写成：

1. 消费线程收到消息
2. 解析、校验、构造参数
3. 启动工作线程
4. 此时不 ack
5. 工作线程执行长任务
6. 工作线程成功或失败后，先发送进度结果
7. 工作线程构造 `ack_callback`
8. 工作线程调用 `connection.add_callback_threadsafe(ack_callback)`
9. `ack_callback` 被投递回 RabbitMQ 消费连接所属线程
10. 消费线程执行 `channel.basic_ack(delivery_tag=...)`

关键点在于第 8 到第 10 步：

- **不是工作线程直接调 ack**
- **而是由 pika 把这个回调安排回消费线程执行**

#### E. 为什么当前项目不需要额外 ack 队列

对于当前项目，只要目标是：

- 成功后 ack
- 失败后也 ack
- ack 必须回到消费线程执行

那么 `connection.add_callback_threadsafe(...)` 已经足够。

因此当前不建议为了 ack 再额外增加：

- 一个新的 Python `Queue`
- 一个新的 ack 管理线程
- 一套额外的 ack 轮询逻辑

因为这样只会增加复杂度，而不会带来明显收益。

当前最小可行且最符合 `pika.BlockingConnection` 习惯的方案就是：

- 工作线程结束后
- 直接调用 `add_callback_threadsafe(...)`
- 把 `ack_callback` 扔回消费线程

#### F. 成功与失败分支的统一做法

为了避免遗漏确认逻辑，工作线程里的 ack 请求应统一放在“任务结束阶段”处理：

- 成功完成任务后，投递 `ack_callback`
- 任务执行异常但失败进度已回传后，也投递 `ack_callback`

这样可以保证：

1. 成功任务最终会被确认
2. 失败任务不会因为漏掉 ack 而长期占住 `prefetch_count=1`
3. 代码行为更一致，不容易出现某个异常分支漏确认的问题

### 6.4 `RabbitMQListener` 的改法

建议在 `connect()` 或 `start_consume()` 中增加：

```python
channel.basic_qos(prefetch_count=1)
```

放置位置要求：

- 应在 `basic_consume()` 之前设置
- 保证 consumer 开始工作前已经带上 QoS 限制

## 七、推荐的确认时机矩阵

| 场景 | 是否 ack | 说明 |
|---|---|---|
| JSON 解析失败 | 是 | 坏消息不应反复重投 |
| `msg_type` 非法 | 是 | 业务无效消息，直接确认并回报失败 |
| 参数校验失败 | 是 | 当前项目走失败回传，不做队列重试 |
| 参数构造失败 | 是 | 同上 |
| 工作线程创建失败 | 是 | 任务未能进入执行态，但消息已处理为失败 |
| 工作线程内任务成功 | 是 | 正常完成后确认 |
| 工作线程内任务异常 | 是 | 失败进度已回传后确认 |

在当前项目里，不建议把这些异常分支改成 `basic_nack(requeue=True)`，否则容易造成：

- 同一坏消息无限重试
- 后端持续收到重复失败
- 单机服务被错误任务卡死

## 八、与当前发送线程模型的关系

当前项目已经有独立的 `SenderThread`：

- 主任务线程通过 `message_queue.put(...)` 推送进度
- `SenderThread` 再独立连接 RabbitMQ 发送到后端队列

这个模型和“完成后 ack”并不冲突，反而是互补的：

1. 消费线程不负责发送进度
2. 工作线程也不直接持有发送连接
3. 发送与消费解耦
4. ack 与进度回传可以分别管理

因此文档建议保持：

- 独立发送线程不变
- 消费线程只负责接收和连接生命周期
- 工作线程只负责执行任务和触发最终 ack 请求

## 九、推荐的最终架构

建议的消费架构如下：

```text
RabbitMQ Server
    |
    v
RabbitMQListener / BlockingConnection / consume thread
    |
    |-- 解析消息
    |-- 参数校验
    |-- 启动工作线程
    |-- 不立即 ack
    |
    v
Worker Thread
    |
    |-- 执行长任务
    |-- 发送进度到本地 Queue
    |-- 成功/失败后请求 ack
    |
    v
connection.add_callback_threadsafe(...)
    |
    v
consume thread 中执行 basic_ack(delivery_tag)

SenderThread
    |
    v
RabbitMQ Server (后端进度队列)
```

同时监听器需要设置：

```text
basic_qos(prefetch_count=1)
```

## 十、结论

结合当前项目的任务模型，推荐的 RabbitMQ 消费策略是：

1. 长任务继续放到后台线程执行，不能阻塞消费线程。
2. 正常任务必须在执行完成后再 ack。
3. 校验失败、解析失败、任务异常等失败分支仍然 ack。
4. ack 不应直接在工作线程里裸调用，而应通过线程安全方式切回消费连接线程执行。
5. 监听器应设置 `prefetch_count=1`，确保上一条消息未确认前，Broker 不投递下一条消息。

这套设计最适合当前项目，因为它同时满足：

- 不阻塞 RabbitMQ 心跳和消费连接
- 不提前确认长任务
- 不让坏消息无限重试
- 不让单实例服务同时堆积多条重任务

## 十一、如何验证确实是“任务完成后才 ack”

为了验证改造后的行为，建议增加一个**专门用于 ack 时机验证的 demo**。

这个 demo 的目标不是验证业务算法，而是验证：

1. 消息在任务执行期间是否保持 `unacked`
2. `basic_ack` 是否在任务完成后才发送
3. `prefetch_count=1` 是否真的阻止了下一条消息提前进入执行态

### 11.1 为什么需要单独的验证 demo

如果只看代码，很难确认运行时是否真的满足“完成后 ack”。

增加专门的 demo 后，可以构造：

- 可重复的长任务
- 清晰的开始/结束/ack 时序日志
- 连续两条消息的顺序行为

这样既能从服务端日志判断，也能结合 RabbitMQ 管理界面观察 `Unacked` 状态。

### 11.2 推荐的 demo 形式

建议增加一个单独的验证 demo，例如：

- `neuracle/demo/rabbitmq_ack_timing_demo.py`

它的职责应当是：

1. 构造一条或两条测试消息
2. 让服务端执行一个**可控的长任务**
3. 在关键时间点输出日志
4. 配合 `prefetch_count=1` 观察消息是否串行执行

这里的“可控长任务”不需要是真实仿真，也可以是：

- 一个故意 `sleep(30)` 的测试任务
- 一个只做少量逻辑但持续数十秒的假任务

这样验证 ack 逻辑会比跑真实建模/仿真更稳定、更可重复。

### 11.3 demo 需要记录的关键日志

建议至少记录以下事件：

1. `received`
   - 消费线程收到消息
2. `worker_started`
   - 工作线程开始执行任务
3. `worker_finished`
   - 工作线程执行完成
4. `ack_scheduled`
   - 工作线程调用 `connection.add_callback_threadsafe(...)`
5. `ack_sent`
   - 消费线程实际执行 `basic_ack(...)`

理想日志顺序应为：

```text
[10:00:00] received task-1
[10:00:00] worker_started task-1
[10:00:30] worker_finished task-1
[10:00:30] ack_scheduled task-1
[10:00:30] ack_sent task-1
```

如果看到：

```text
[10:00:00] received task-1
[10:00:00] ack_sent task-1
[10:00:30] worker_finished task-1
```

就说明仍然是提前 ack。

### 11.4 推荐的验证步骤

建议按下面步骤验证：

1. 启动 RabbitMQ 服务端和 management 界面
2. 启动本项目的监听服务
3. 确保监听器已设置 `prefetch_count=1`
4. 运行 `rabbitmq_ack_timing_demo.py`
5. 连续发送两条长任务消息
6. 同时观察：
   - 服务端日志
   - RabbitMQ 管理界面的 `Unacked`

### 11.5 预期现象

如果实现正确，应当看到：

1. 第一条消息开始执行后，队列 `Unacked = 1`
2. 在第一条任务完成前，第二条消息不会开始执行
3. 第一条任务完成后，才出现 `ack_sent`
4. 第一条 `ack_sent` 之后，第二条消息才开始进入执行态

也就是说，行为上应表现为：

- “执行期间保持未确认”
- “完成后确认”
- “未确认期间不接收下一条”
