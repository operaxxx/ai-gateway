# RCA：流式上游错误触发 ResponseNotRead，日志丢失原始错误信息

- **日期**：2026-09-15
- **模式**：dev-fix（默认档）
- **状态**：已修复并回归验证

## Triage

- **Symptom**：流式请求异常时，服务日志只有 `"stream 未预期异常", error_message: "Attempted to access streaming response content, without having called read()."`，看不到上游的原始错误（type / status_code / message），也没有 traceback。
- **Expected**：上游返回非 2xx 时，应产出结构化 error 事件（含上游真实 type/status_code/message）；即使走到未预期异常兜底，日志也应带完整堆栈。
- **Repro 起点**：POST /v1/chat（stream=true）打到上游错误响应（如无效模型名触发 400）。
- **Environment**：Python 3.13 + FastAPI + httpx（`client.stream()` 流式中继模式）。
- **First seen**：控制台页面流式调用失败时（2026-09-15）。
- **Severity**：影响功能（错误不可观测，排障困难）。

## 复现

单测复现（tests/test_stream_error_path.py）：

- MockTransport handler 返回 `httpx.Response(status, content=iter([body]))`——**惰性 content**，构造"流式响应体未读"状态。
- 断言：适配器应 yield 单个 error 事件，携带上游真实 type/status_code/message。
- 修复前：实际抛出 `httpx.ResponseNotRead`，无 error 事件 → 测试 RED（4/4）。

**关键复现技巧**：`httpx.Response(500, text=...)` 或 `json=...` 会在构造时立即设置 `_content`，`.json()` 不报错，无法复现"未读"状态；必须用 `content=iter([body])`。

## 假设

- **H1（证实）**：`ResponseNotRead` 异常穿透。旧代码在 `client.stream()` 的 with 块外调 `raise_for_status()`，except 分支里 `from_http_response` → `errors.py` 的 `_safe_json` 调 `response.json()`——但流式响应体未 read，抛 `httpx.ResponseNotRead`。
- **H1 的放大因素**：`ResponseNotRead` 继承 `httpx.StreamError` 而**不是** `httpx.HTTPError`，适配器的 `except httpx.HTTPError` 捕不到，一路穿透到 Gateway 的未预期异常兜底；而兜底当时用的是 `logger.error`（无 exc_info），JsonFormatter 拿不到 traceback → 日志既丢上游信息又丢堆栈。

## 根因链

1. 流式错误路径在 **with 块外**处理错误响应（body 必然未读）。
2. `_safe_json` 直接访问 `response.json()`，未考虑流式未读态。
3. `ResponseNotRead` ≠ `HTTPError`，异常分类体系漏接。
4. Gateway 兜底 `logger.error` 不带 exc_info，日志无 tb 字段。

## 修复

1. **两适配器 stream()**（anthropic_adapter.py / responses_adapter.py）：
   - 错误处理移入 with 块内：`if response.is_error: response.read()`（with 块内 read 幂等且官方支持），再 `from_http_response` 构造 error 事件 yield 后 return；
   - 删除 `raise_for_status()` 与对应的 `except httpx.HTTPStatusError` 分支；
   - `except httpx.HTTPError` 保留兜网络层错误（连接/超时）。
2. **gateway.py**：`complete()` / `stream()` 未预期异常 `logger.error` → `logger.exception`，JsonFormatter 的 `tb` 字段随之输出完整堆栈。

外科手术边界：只动错误路径，不改正常流式翻译逻辑。

## 防御与回归

- tests/test_stream_error_path.py：{AnthropicAdapter, ResponsesAdapter} × {400, 500} 参数化，断言 error 事件携带上游真实错误三要素；修复前 RED（4/4）→ 修复后 GREEN，stash/pop 交叉验证。
- **旧测试为何漏测**：test_stream_mock.py 的 `error_client()` 用 `text=` 构造响应，`_content` 已就绪，绕过了未读态——mock 流式错误必须用惰性 `content=iter([...])`。

## 验证

- 新增回归 4 用例：RED → GREEN → stash RED / pop GREEN。
- 全量 pytest：86 通过（修复时点）；后续批次（思考开关 / 结构化输出）增至 94，持续全绿。
- 真实端到端：无效请求触发上游 400，日志出现结构化 error 事件；未预期异常日志带 tb 字段。

## 预防清单

- [x] 流式错误处理一律在 `client.stream()` with 块内完成 `response.read()` 后再解析。
- [x] 涉及未预期异常的兜底日志一律用 `logger.exception`（带堆栈）。
- [x] mock 上游流式错误时用惰性 content，禁止 `text=`/`json=` 构造。
- [ ] 备忘：任何新的 `except httpx.XXX` 都要核对 httpx 异常继承树（StreamError 与 HTTPError 是平级分支）。
