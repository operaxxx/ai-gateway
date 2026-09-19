# 模型级别限流技术方案（多级限流防护体系）

> 状态：已实现并验证（2026-09-19）
> 范围：在既有 IP 滑动窗口限流之上叠加模型级令牌桶限流，形成两级防护

---

## 1. 背景与目标

### 1.1 现状与不足

系统原有基于客户端 IP 的滑动窗口限流（`gateway/ratelimit.py` 的 `SlidingWindowLimiter`，在 `server.py` 的 `_request_id_middleware` 中执行），配置 `RATE_LIMIT_RPM` / `RATE_LIMIT_WINDOW_S`。该机制只回答"这个客户端是否刷量"，无法回答：

- **配额保护**：`deepseek-v4-pro` 成本高、上游配额少，需要单独限制；`deepseek-v4-flash` 便宜可以放宽
- **成本失控**：单个模型被滥用会耗尽整个网关的上游预算，即使每个客户端 IP 都没触发 IP 限流
- **公平调度**：N 个客户端分散调用昂贵模型时，总量仍可能压垮上游配额

### 1.2 目标

1. 增加 per-model 限流能力，不同模型独立配额、独立算法选择
2. 与 IP 限流协同工作：任一级拒绝即 429，客户端可通过错误体区分来源
3. 不引第三方依赖（零依赖原则），单实例内存态实现
4. 向后兼容：不配置模型限流时，行为与现网完全一致

---

## 2. 业界限流策略调研

### 2.1 四种主流算法对比

| 算法 | 原理 | 精确性 | 突发支持 | 内存 | 复杂度 | 典型场景 |
|---|---|---|---|---|---|---|
| 固定窗口计数器 | 每个固定窗口（如每分钟）独立计数 | 差：窗口边界突刺（两窗口交界处可达 2 倍限值） | 否 | O(1) | 低 | 粗粒度配额统计 |
| 滑动窗口（日志型） | 记录窗口内每次命中时间戳，随时间滑动淘汰 | 精确：任意时刻回看真实窗口 | 否 | O(N)（N=窗口内命中数） | 中 | 精确次数限制（如 IP 防刷） |
| 漏桶 | 请求入桶，恒定速率流出 | 严格匀速，平滑到每一毫秒 | 否（突发被完全抹平） | O(1) | 中 | 保护带宽/硬实时下游 |
| 令牌桶 | 恒定速率往桶里放令牌，请求消耗令牌 | 平均速率精确 | 是（桶容量即突发上限） | O(1) | 中 | 允许突发 + 平均限速 |

### 2.2 LLM 网关场景分析

LLM 调用流量有两个显著特征：

1. **天然突发**：人类/Agent 使用模式是"连续几轮对话 → 停顿 → 再对话"。滑动窗口/漏桶会把自然突发错杀（用户连发 3 条消息就被拒），令牌桶则允许 burst 内瞬间通过、空闲期自动攒额度
2. **上游配额语义匹配**：LLM 供应商自身的 rate limit（RPM + TPM）本质就是令牌桶语义，网关侧用同构算法做前置保护，减少无效的上游 429 往返

结论：
- **IP 级（防刷量）**：保留滑动窗口。防刷恰恰要"精确不允许钻空子"，拒绝突刺是优点，且被拒请求不占名额的语义利于快速恢复
- **模型级（保配额）**：采用令牌桶。允许自然突发、平均速率受控，与上游配额语义同构

---

## 3. 多级限流架构设计

### 3.1 请求路径与两级检查点

```
Client
  │
  ▼
┌─────────────────────────────────────────────────┐
│ server.py _request_id_middleware                 │
│   ┌───────────────────────────────┐             │
│   │ 一级 IP 限流（滑动窗口）        │  被拒 → 429 scope=ip
│   │ SlidingWindowLimiter          │             │
│   │ 粗筛：尽早拒绝，省下游资源      │             │
│   └───────────────┬───────────────┘             │
└───────────────────┼─────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ server.py chat() 路由                            │
│   ┌───────────────────────────────┐             │
│   │ 二级模型限流（令牌桶）          │  被拒 → 429 scope=model
│   │ TokenBucketLimiter            │             │
│   │ 细筛：model 粒度，LLM 调用前    │             │
│   └───────────────┬───────────────┘             │
└───────────────────┼─────────────────────────────┘
                    ▼
            Gateway → 上游 LLM
```

协同语义：
- **串联执行**：IP 通过才进入 chat 路由检查模型桶；中间件被拒时路由不执行，模型令牌不被消耗
- **任一被拒即 429**，两级计数器独立（`rate_limited_total` / `model_rate_limited_total`），互不污染
- **检查点位置**：模型级在 `req.model not in MODEL_ROUTES` 校验之后、结构化输出预检之前——LLM 调用链上最早的 model 粒度可判定点

### 3.2 Limiter 统一协议

```python
class Limiter(Protocol):
    def check(self, key: str, now: float | None = None) -> tuple[bool, float]: ...
```

- `typing.Protocol` 鸭子类型：现有 `SlidingWindowLimiter` 无需任何改造即满足协议，新增算法零侵入
- `check()` 语义统一：放行时登记/扣减，拒绝时**不占名额**（两种算法一致），`retry_after_s >= 1`
- `now` 参数可注入，测试无需 mock 时钟
- 扩展点：未来加租户级/全局级限流，实现该协议即可插入任意层级

### 3.3 TokenBucketLimiter 算法细节

状态（per-key，key=model 名）：`[tokens, last_refill_t]`

```python
elapsed = now - last_refill_t
tokens  = min(capacity, tokens + elapsed * rate)   # 惰性填充，封顶桶容量
if tokens >= 1.0:
    tokens -= 1.0        # 消耗 1 个令牌
    return True, 0.0
retry_after = ceil((1.0 - tokens) / rate)          # 攒够 1 个令牌所需时间
return False, max(1.0, retry_after)
```

关键设计：

| 设计点 | 决策 | 理由 |
|---|---|---|
| 填充方式 | **惰性填充**（check 时按时间差补），无后台线程 | 零常驻开销；空闲期零 CPU；与项目"零依赖 + stdlib"原则一致 |
| 初始状态 | 满桶（tokens = burst） | 冷启动友好；首次部署不会立即误拒 |
| 拒绝是否扣令牌 | 不扣 | 与滑动窗口"被拒不占名额"语义统一，恢复可预期 |
| Retry-After | `ceil((1-tokens)/rate)` | 攒够 1 个令牌的精确等待时间，客户端重试不浪费 |
| 并发安全 | `threading.Lock` 包住整个 check | uvicorn 线程池并发写；临界区极短（纯内存运算），无性能问题 |
| 内存回收 | 无 TTL | key = model 名，数量受 `MODEL_ROUTES`（当前 2 个）约束，无泄漏风险 |

对比漏桶：漏桶会把突发完全抹平（用户连发 3 条消息强制排队/拒绝），不符合 LLM 交互模式；令牌桶"平均受限 + 突发放行"是 LLM 场景的最优解。

---

## 4. 实现细节

### 4.1 配置格式

环境变量 `MODEL_RATE_LIMITS`（JSON 字符串，`.env` 兜底，显式 export 优先——与全项目配置语义一致）：

```bash
export MODEL_RATE_LIMITS='{"deepseek-v4-pro": {"rpm": 10, "burst": 20}, "deepseek-v4-flash": {"rpm": 60}}'
```

- `rpm`：每分钟平均速率上限（令牌填充速率 = rpm/60 每秒）
- `burst`：桶容量（允许的瞬时突发上限）；**省略时默认 = rpm**
- 未设置/空字符串/非法 JSON → `{}`（禁用模型限流，**fail-open**）
- fail-open 理由：配置错误只应降级为"无限流"，不应阻断网关启动或误封全部模型；同时记 WARNING 日志暴露问题
- 单条目非法（rpm/burst 非正数、非数字）→ 跳过该条目，其余照常生效

### 4.2 错误体与响应头（两级同构）

两级限流返回**完全同构**的错误体，仅 `scope` 与附加字段不同：

```json
// 一级：IP 限流（中间件）
{"error": {
  "category": "rate_limited",
  "type": "rate_limit_error",
  "scope": "ip",
  "message": "请求过于频繁（限流 6 次/4s 窗口），请稍后重试",
  "retryable": true
}}
// 二级：模型限流（路由内）
{"error": {
  "category": "rate_limited",
  "type": "rate_limit_error",
  "scope": "model",
  "model": "deepseek-v4-flash",
  "message": "模型 deepseek-v4-flash 调用频率超限（12 rpm，突发上限 3），请稍后重试",
  "retryable": true
}}
```

两者都带 `Retry-After` 头（整数秒）。实现上模型级用 `JSONResponse` 而非 `HTTPException`——FastAPI 会把 HTTPException 包成 `{"detail": {...}}`，破坏顶层 `{"error": {...}}` 结构，导致客户端需要两套解析。

### 4.3 指标扩展

`gateway/metrics.py` 新增计数器：

| 计数器 | 含义 | 告警用法 |
|---|---|---|
| `rate_limited_total`（既有） | IP 级拒绝数 | 突增 = 单客户端刷量/CC 攻击 |
| `model_rate_limited_total`（新增） | 模型级拒绝数 | 突增 = 该模型配额不足或被集中滥用 |

中间件出口统一统计 `requests_total/errors_total/status_counts/http_latency`（路由内 return 的 429 也经过 `call_next` 返回，天然覆盖）；路由内只记模型级特有计数器，**避免双计**。

### 4.4 文件改动清单

| 文件 | 改动 |
|---|---|
| `gateway/ratelimit.py` | 新增 `Limiter` Protocol、`TokenBucketLimiter`、`from_env_for_models()`；`from_env` 返回类型注解改 `Limiter \| None` |
| `server.py` | `model_limiters` 全局单例；`chat()` 内二级限流检查；IP 429 错误体补 `scope:"ip"` |
| `gateway/metrics.py` | 新增 `model_rate_limited_total` 计数器 |
| `tests/test_ratelimit_metrics.py` | TokenBucketLimiter 单测 + 配置解析测试 + 多级协同集成测试 |
| `scripts/verification/verify_rate_limit.py` | 新增模型级限流端到端验证段 |

---

## 5. 测试方法

### 5.1 测试矩阵

| 层级 | 覆盖内容 | 位置 |
|---|---|---|
| 单元：令牌桶 | 突发消耗至耗尽、Retry-After 数学（rate=1/s→1s；rate=0.5/s→2s）、惰性填充恢复、容量封顶（空闲 1h 不超 capacity）、per-key 隔离、拒绝不扣令牌、4 线程并发恰好放行 burst 个、非法参数 | `tests/test_ratelimit_metrics.py::TestTokenBucketLimiter` |
| 单元：配置解析 | 默认禁用、空串禁用、合法 JSON、burst 省略默认=rpm、非法 JSON fail-open、非对象 JSON fail-open、非法条目跳过 | `::TestFromEnvForModels` |
| 集成：多级协同 | (a) IP 未触限模型被拒→scope=model，IP 计数器为 0；(b) IP 先拒→scope=ip，模型检查未执行（令牌未消耗）；(c) 两级都通过；(d) per-model 隔离 | `::TestMultiLevelRateLimit` |
| 端到端 | mock 上游 + 真实网关子进程：突发精确触发、scope=model 响应、per-model 隔离、分级计数器、令牌填充恢复 | `scripts/verification/verify_rate_limit.py` 步骤 7-12 |
| 回归 | 不设 `MODEL_RATE_LIMITS` 时既有 IP 限流测试/验证脚本行为不变 | 既有测试全部保留并通过 |

### 5.2 运行方式

```bash
# 单元 + 集成
uv run python -m pytest tests/test_ratelimit_metrics.py -v

# 端到端（拉起 mock + 网关子进程，报告落盘 reports/rate_limit.json）
uv run python scripts/verification/verify_rate_limit.py
```

### 5.3 实测结果

- 单元/集成：30 passed（含新增 19 个用例）
- 端到端：27/27 PASS（IP 段 14 项 + 模型段 13 项）

---

## 6. 性能评估

| 维度 | 评估 |
|---|---|
| 请求路径开销 | 二级检查 = 1 次 dict.get + 1 次 Lock 内 O(1) 运算，微秒级；相对 LLM 上游往返（百 ms 级）可忽略 |
| 锁粒度 | 单把锁保护桶字典。临界区纯内存运算（无 IO），2 个模型 + 线程池 4 并发的规模下无争用热点；若未来模型数/并发量大，可改 per-key 分段锁（预留优化点，当前 YAGNI） |
| 后台开销 | 惰性填充 = 零后台线程、零定时器；空闲期完全零 CPU |
| 内存上界 | O(模型数)。key=model 名受路由表约束（当前 2 个），每桶 2 个 float，无泄漏路径 |
| 与 IP 限流叠加延迟 | IP 检查（中间件）+ model 检查（路由）共 2 次锁内 O(1) 运算；本地实测 P50 不可测出差异（远小于网络抖动） |

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 配置错误导致全部模型被误封 | fail-open：非法 JSON/条目 → 该部分禁用 + WARNING 日志，不阻断启动 |
| 单实例非共享状态 | 与既有 IP 限流一致的明确边界：水平扩容 N 个实例，实际 QPS 上限按实例数放大（`N × burst`）。文档明示；如需全局限额，后续引入集中存储（Redis）时只需替换 Limiter 实现，协议不变 |
| 满桶初始化的启动瞬时压力 | 部署后允许 burst 个瞬时请求。对冷启动场景是特性不是缺陷；如需保守可配小 burst |
| 模型名不受控导致桶字典膨胀 | 检查点在 `MODEL_ROUTES` 校验之后执行，非法模型名到不了限流器；桶 key 集合恒 ≤ 支持的模型数 |
| 客户端不识别 scope 字段 | 向后兼容：scope 是新增字段，既有字段（category/type/message/retryable）语义不变，旧客户端零影响 |

---

## 8. 运维手册

```bash
# 查看两级限流计数
curl -s http://localhost:8000/v1/metrics | jq '.counters | {rate_limited_total, model_rate_limited_total}'

# 调整模型配额（改环境变量后重启生效）
MODEL_RATE_LIMITS='{"deepseek-v4-pro":{"rpm":10,"burst":20}}' uv run uvicorn server:app --port 8000

# 快速验证某模型配额（burst=2）
MODEL_RATE_LIMITS='{"deepseek-v4-flash":{"rpm":2,"burst":2}}' uv run uvicorn server:app --port 8000
# 连续 3 次 curl /v1/chat，第 3 次应 429 且 scope=model
```
