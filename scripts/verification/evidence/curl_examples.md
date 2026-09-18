# 六大功能 curl 复现示例

对 `scripts/verification/run_all.py` 六大模块的**真实服务复现命令**（mock 验证之外的补充证据）。
前置：网关已启动并配置真实密钥（默认 `http://localhost:8000`）。

```bash
export BASE=http://localhost:8000
```

## 1. 流式输出（streaming）

SSE 逐事件返回；`-N` 关闭 curl 缓冲。加 `"thinking": true` 可看到 `reasoning` 通道的思考增量。

```bash
curl -N $BASE/v1/chat -H "Content-Type: application/json" -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "用一句话解释什么是API网关"}],
  "stream": true,
  "thinking": true
}'
# 预期：event: start → 多条 event: delta（channel=reasoning/text）→ event: done
#       done 含 usage / stop_reason / ttft_ms / elapsed_ms
# 响应头含 Content-Type: text/event-stream、Cache-Control: no-cache、X-Accel-Buffering: no
```

## 2. 结构化输出（structured_output）

`response_format` 为裸 JSON Schema 字典。非流式返回 `structured_output` 对象（违反 schema 时 422）；
流式同开时 delta 为 JSON 增量，校验结论随 `done.validation` 下发。

```bash
# 非流式
curl -s $BASE/v1/chat -H "Content-Type: application/json" -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "介绍北京的天气"}],
  "response_format": {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temp": {"type": "integer"}},
    "required": ["city", "temp"],
    "additionalProperties": false
  }
}'
# 预期：响应含 "structured_output": {...} 且通过 schema 校验，附 elapsed_ms

> **实测注意**：flash 走 Anthropic 协议的 tool_use 模式，因 DeepSeek thinking 模式不支持强制
> `tool_choice`，真实模型**偶发**不调用工具而直接文本回答 → 422（或流式 `done.validation.ok=false`），
> 重试即可。需要协议级强制时用 pro 模型（Responses json_schema，实测稳定）：

```bash
curl -s $BASE/v1/chat -H "Content-Type: application/json" -d '{
  "model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "介绍北京的天气"}],
  "response_format": {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temp": {"type": "integer"}},
    "required": ["city", "temp"]
  }
}'
```

# 流式 + 结构化同开（校验结论在 done.validation）
curl -N $BASE/v1/chat -H "Content-Type: application/json" -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "介绍北京的天气"}],
  "stream": true,
  "response_format": {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temp": {"type": "integer"}},
    "required": ["city", "temp"]
  }
}'
# 预期：delta 逐块为 JSON 增量；done.validation.ok=true 且 parsed 为完整对象
```

## 3. 模板引用（prompt_template）

创建模板（`{{变量}}` / `{% if %}` / `{% for %}`）→ 渲染预览 → chat 内联引用（服务端渲染为 system）。

```bash
# 创建模板（服务启动时会自动播种 translator 模板，此处新建带条件/循环的示例）
curl -s -X POST $BASE/v1/prompts -H "Content-Type: application/json" -d '{
  "id": "review-guide",
  "name": "评审助手",
  "content": "你是{{role}}。{% if formal %}请使用敬语。{% else %}轻松一点。{% endif %}{% for r in rules %}- {{r}}\n{% endfor %}任务：{{task}}"
}'

# 渲染预览（不调 LLM）：条件 true 分支 + 循环逐项
curl -s -X POST $BASE/v1/prompts/review-guide/render -H "Content-Type: application/json" -d '{
  "version": "latest",
  "variables": {"role": "资深工程师", "formal": true,
                "rules": ["简洁", "准确"], "task": "评审以下代码"}
}'
# 缺变量 → 400 missing_variables；语法错误 → 400

# chat 内联引用：variables 服务端渲染后作为 system 发给上游
curl -s $BASE/v1/chat -H "Content-Type: application/json" -d '{
  "model": "deepseek-v4-flash",
  "prompt": {"id": "review-guide", "version": "latest",
             "variables": {"role": "资深工程师", "formal": true,
                           "rules": ["简洁", "准确"], "task": "评审以下代码"}},
  "messages": [{"role": "user", "content": "请开始"}]
}'
# version 钉死历史版本（如 1）后渲染结果不可变；prompt 与自带 system 消息互斥（同用 400）
```

## 4. 可观测数据（observability）

指标快照 + 健康检查；每次响应携带 `elapsed_ms`，JSON 日志含全链路 `request_id`（`LOG_FILE` 落盘）。

```bash
curl -s $BASE/health      # {"status":"ok"}
curl -s $BASE/v1/metrics  # counters.requests_total / llm_calls_total / errors_total / retries_total
                          # rate_limited_total；status_counts / llm_latency 分位数 / llm_ttft 为顶层字段
curl -s $BASE/v1/chat -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['elapsed_ms'], 'ms')"
# 前后各抓一次 /v1/metrics 对比，requests_total 增量与请求次数一致
```

## 5. 重试机制（retry）

重试发生在网关内部（仅 network / server / 429 / 409 错误，指数退避并尊重 `Retry-After`），
curl 层表现为"最终成功"。观测手段是 `/v1/metrics` 的 `retries_total` 与 JSON 日志：

```bash
# 上游抖动（500/429/超时）时请求仍返回 200；对比前后 retries_total 增量即重试次数
curl -s $BASE/v1/metrics | python3 -c "import json,sys; print(json.load(sys.stdin)['counters']['retries_total'])"
# ...期间发起业务请求...
curl -s $BASE/v1/metrics | python3 -c "import json,sys; print(json.load(sys.stdin)['counters']['retries_total'])"

# JSON 日志中可见 WARNING 级"上游可重试错误"记录（含 attempt / backoff / request_id）
tail -f gateway.log | grep -i retry
```

## 6. 限流（rate_limit）

需以限流配置启动（per 客户端 IP 滑动窗口），超限返回 429 + `Retry-After` 头：

```bash
RATE_LIMIT_RPM=6 RATE_LIMIT_WINDOW_S=60 uvicorn server:app --port 8000

# 连发 12 个请求：前 6 个 200，第 7 个起 429
for i in $(seq 1 12); do
  curl -s -o /dev/null -w "%{http_code}\n" $BASE/v1/chat \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}'
done

# 查看 429 响应体与 Retry-After 头
curl -si $BASE/v1/chat -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}' | head -20
# 预期：HTTP/1.1 429、Retry-After: N、body 含 "error": {"type": "rate_limit_error", "retryable": true}
```

> 注：422（schema 违反）与上游故障注入路径依赖可控的上游行为，已在 mock 验证
> （`run_all.py` 的 structured_output / retry 模块）中覆盖；本文 curl 示例覆盖真实上游下的可达路径。
