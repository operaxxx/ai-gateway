/* AI Gateway 控制台：原生 JS，无框架、无构建、无 CDN。
 *
 * 模块分区：
 *   1. 常量与状态   2. 工具   3. API 层（fetch 封装 + SSE 解析）
 *   4. localStorage 5. 模板编辑器   6. 变量表单与渲染预览
 *   7. 调用（流式/非流式）8. 日志   9. 历史与对比   10. init 与事件绑定
 *
 * 安全约定：所有用户数据（模板正文、渲染结果、模型输出、变量值、日志 JSON）
 * 一律经 textContent / createElement 进 DOM，不用 innerHTML 拼接。
 */
(() => {
'use strict';

// ==================== 1. 常量与状态 ====================

const LS_HISTORY_KEY = 'ai-gateway.history.v1';
const HISTORY_CAP = 30;         // 历史上限（localStorage）
const LOG_CAP = 200;            // 内存日志上限
const SNIPPET = 500;            // 日志中文本截断长度
const RESULT_SNIPPET = 10000;   // 历史中文本截断长度

const state = {
  models: [],
  templates: [],
  editor: {                      // 模板编辑器状态机
    mode: null,                  // null | 'new' | 'edit'
    tplId: null,
    versions: [],                // 编辑模式的版本列表
    viewingVersion: null,        // null=latest（可编辑）；数字=旧版本（只读）
    baselineContent: '',         // latest 原文（判断是否有改动）
    currentVars: [],             // latest 的服务端变量
  },
  varDefs: [],                   // 变量表单当前变量名列表
  history: [],
  logs: [],
  seq: 0,
  compareSel: [],                // 待对比的历史 id（最多 2）
  abortCtrl: null,
  sending: false,
};

const STREAM_LABELS = {
  idle: '空闲', pending: '请求中…', streaming: '流式接收中…', done: '完成', error: '出错',
};

// ==================== 2. 工具 ====================

const $ = (sel) => document.querySelector(sel);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtTime(d = new Date()) {
  const p = (n, w = 2) => String(n).padStart(w, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
}

function truncate(s, n) {
  if (typeof s !== 'string') return s;
  return s.length > n ? s.slice(0, n) + `…（共 ${s.length} 字）` : s;
}

// 兼容两种错误形态：{detail: str|{message,missing}} 与 {error:{...}}
function extractError(status, body) {
  if (body == null) return `HTTP ${status}`;
  if (body.detail !== undefined) {
    const d = body.detail;
    if (typeof d === 'string') return d;
    let msg = d.message || JSON.stringify(d);
    if (d.missing) msg += `（缺失变量: ${d.missing.join(', ')}）`;
    return msg;
  }
  if (body.error) return body.error.message || JSON.stringify(body.error);
  return JSON.stringify(body);
}

// 变量输入：能 JSON.parse 就用解析值（数字/布尔/数组/对象），否则原样字符串
function tryParseScalar(s) {
  const t = s.trim();
  if (t === '') return { empty: true, value: null };
  try { return { empty: false, value: JSON.parse(t) }; }
  catch { return { empty: false, value: s }; }
}

// 草稿期粗提取 {{var}}；{% if %}/{% for %} 引用的变量要等服务端 extract_variables
function draftVariables(content) {
  const re = /\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g;
  const out = [];
  let m;
  while ((m = re.exec(content)) !== null) {
    if (!out.includes(m[1])) out.push(m[1]);
  }
  return out;
}

// ==================== 3. API 层 ====================

async function api(method, path, body, opts = {}) {
  const t0 = performance.now();
  let status = null;
  let data = null;
  let networkErr = null;
  try {
    const resp = await fetch(path, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: opts.signal,
    });
    status = resp.status;
    const text = await resp.text();
    if (text) {
      try { data = JSON.parse(text); } catch { data = text; }
    }
  } catch (e) {
    networkErr = e;
  }
  const duration_ms = performance.now() - t0;
  const ok = networkErr === null && status !== null && status >= 200 && status < 300;
  addLog({
    method, path, status, duration_ms, ok,
    summary: ok && opts.summarize ? opts.summarize(data) : '',
    request: body !== undefined ? body : null,
    response: networkErr ? null : data,
    error: networkErr ? String(networkErr) : null,
  });
  if (networkErr) throw networkErr;
  return { ok, status, data, duration_ms };
}

function chatSummary(data) {
  if (!data || data.error) return '';
  const u = data.usage || {};
  const parts = [`in=${u.input_tokens ?? '?'}`, `out=${u.output_tokens ?? '?'}`];
  if (data.stop_reason) parts.push(String(data.stop_reason));
  if (data.elapsed_ms != null) parts.push(`elapsed=${data.elapsed_ms}ms`);
  return parts.join(' · ');
}

// SSE 解析器：chunk 可能在任意位置切断事件块，必须缓冲到 \n\n 再切分
function createSSEParser(onEvent) {
  let buf = '';
  return {
    feed(chunk) {
      buf += chunk.replace(/\r\n/g, '\n');
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = 'message';
        let data = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) data += (data ? '\n' : '') + line.slice(5).trimStart();
        }
        if (data) {
          try { onEvent(event, JSON.parse(data)); } catch { /* 坏帧忽略 */ }
        }
      }
    },
  };
}

// 流式调用：需 POST，不能用 EventSource，用 fetch + ReadableStream 手动解析
async function streamChat(payload, { onEvent, signal }) {
  const resp = await fetch('/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!resp.ok || !resp.body) {
    // 非 2xx 时 body 是 JSON 错误体，不是 SSE
    const text = await resp.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }
    return { ok: false, status: resp.status, data };
  }
  const parser = createSSEParser(onEvent);
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parser.feed(dec.decode(value, { stream: true }));
  }
  parser.feed(dec.decode());   // 冲掉残余缓冲（最后一帧后可能无空行）
  return { ok: true, status: resp.status };
}

// ==================== 4. localStorage ====================

function loadHistory() {
  try {
    const raw = localStorage.getItem(LS_HISTORY_KEY);
    state.history = raw ? (JSON.parse(raw) || []) : [];
  } catch { state.history = []; }
}

function saveHistory() {
  try {
    localStorage.setItem(LS_HISTORY_KEY, JSON.stringify(state.history));
  } catch {
    // 容量超限：删最旧一半重试一次
    state.history = state.history.slice(0, Math.ceil(state.history.length / 2));
    try { localStorage.setItem(LS_HISTORY_KEY, JSON.stringify(state.history)); } catch { /* 放弃 */ }
  }
}

function appendHistory(entry) {
  state.history.unshift(entry);
  if (state.history.length > HISTORY_CAP) state.history.length = HISTORY_CAP;
  saveHistory();
  renderHistory();
}

// ==================== 5. 模板编辑器 ====================

function showTplMsg(text, isErr) {
  const msg = $('#tpl-msg');
  msg.hidden = false;
  msg.textContent = text;
  msg.className = 'msg ' + (isErr ? 'err' : 'ok');
}

async function refreshTemplates() {
  const r = await api('GET', '/v1/prompts');
  state.templates = r.ok ? r.data.prompts : [];
  renderTemplateList();
}

function renderTemplateList() {
  const list = $('#tpl-list');
  list.textContent = '';
  const kw = $('#tpl-search').value.trim().toLowerCase();
  const tpls = state.templates.filter(t =>
    !kw || t.id.toLowerCase().includes(kw) || t.name.toLowerCase().includes(kw));
  if (!tpls.length) {
    list.append(el('li', 'empty', kw ? '无匹配模板' : '暂无模板，点击「新建模板」创建'));
    return;
  }
  for (const t of tpls) {
    const li = el('li');
    if (state.editor.mode === 'edit' && state.editor.tplId === t.id) li.classList.add('active');
    li.append(el('span', 'tpl-name', t.name || t.id));
    li.append(el('span', 'tpl-id', t.id));
    li.append(el('span', 'tpl-ver', `v${t.latest_version}`));
    li.addEventListener('click', () => openTemplate(t.id));
    list.append(li);
  }
}

async function openTemplate(id) {
  const detail = await api('GET', `/v1/prompts/${id}`);
  const latest = await api('GET', `/v1/prompts/${id}/versions/latest`);
  if (!detail.ok || !latest.ok) {
    showTplMsg(`加载模板失败: ${extractError(detail.status, detail.data)}`, true);
    return;
  }
  state.editor = {
    mode: 'edit',
    tplId: id,
    versions: detail.data.versions,
    viewingVersion: null,
    baselineContent: latest.data.content,
    currentVars: latest.data.variables,
  };
  $('#tpl-new-fields').hidden = true;
  $('#tpl-edit-meta').hidden = false;
  $('#tpl-edit-title').textContent = `${id} · ${detail.data.name}`;
  $('#btn-tpl-delete').hidden = false;

  const sel = $('#tpl-version-select');
  sel.textContent = '';
  const optLatest = el('option', null, 'latest（可编辑）');
  optLatest.value = 'latest';
  sel.append(optLatest);
  for (const v of state.editor.versions) {
    const opt = el('option', null, `v${v.version}（只读）`);
    opt.value = String(v.version);
    sel.append(opt);
  }
  sel.value = 'latest';

  $('#tpl-content').value = latest.data.content;
  setVarDefs(latest.data.variables);
  updateVersionHint(false);
  renderTemplateList();
  syncPromptRefControls();
}

function editorNew() {
  state.editor = { mode: 'new', tplId: null, versions: [], viewingVersion: null, baselineContent: '', currentVars: [] };
  $('#tpl-new-fields').hidden = false;
  $('#tpl-edit-meta').hidden = true;
  $('#tpl-version-hint').hidden = true;
  $('#tpl-id').value = '';
  $('#tpl-name').value = '';
  $('#tpl-desc').value = '';
  $('#tpl-content').value = '';
  $('#tpl-content').readOnly = false;
  $('#tpl-content').placeholder = 'Jinja2 模板：{{var}} 占位，支持 {% if %}、{% for %}、{{ v | default("兜底") }}';
  $('#btn-tpl-delete').hidden = true;
  $('#btn-tpl-save').hidden = false;
  $('#btn-tpl-save').disabled = false;
  $('#btn-tpl-save').textContent = '保存（创建 v1）';
  setVarDefs([]);
  $('#tpl-msg').hidden = true;
  renderTemplateList();
  syncPromptRefControls();
  $('#tpl-id').focus();
}

function resetEditor() {
  state.editor = { mode: null, tplId: null, versions: [], viewingVersion: null, baselineContent: '', currentVars: [] };
  $('#tpl-new-fields').hidden = true;
  $('#tpl-edit-meta').hidden = true;
  $('#tpl-version-hint').hidden = true;
  $('#tpl-content').value = '';
  $('#tpl-content').readOnly = false;
  $('#btn-tpl-delete').hidden = true;
  $('#btn-tpl-save').hidden = true;
  setVarDefs([]);
  $('#tpl-msg').hidden = true;
  renderTemplateList();
  syncPromptRefControls();
}

// isDraft: 是否处于可编辑的 latest 草稿视图
function updateVersionHint(isDraft) {
  const hint = $('#tpl-version-hint');
  const saveBtn = $('#btn-tpl-save');
  const meta = state.templates.find(t => t.id === state.editor.tplId);
  const latestV = meta ? meta.latest_version : (state.editor.versions[0]?.version ?? 1);
  const readOnly = !isDraft;
  $('#tpl-content').readOnly = readOnly;
  saveBtn.hidden = false;
  saveBtn.disabled = readOnly;
  saveBtn.textContent = isDraft ? `保存（发布 v${latestV + 1}）` : '保存（仅最新版可编辑）';
  hint.hidden = false;
  hint.textContent = isDraft
    ? `基于 v${latestV} 编辑，保存将创建新版本（历史版本不可变）`
    : `正在查看 v${state.editor.viewingVersion}（历史版本不可变，只读）`;
}

async function saveTemplate() {
  const content = $('#tpl-content').value;
  if (state.editor.mode === 'new') {
    const id = $('#tpl-id').value.trim();
    const name = $('#tpl-name').value.trim();
    if (!id || !name) { showTplMsg('请填写 id 和名称', true); return; }
    const r = await api('POST', '/v1/prompts',
      { id, name, description: $('#tpl-desc').value.trim(), content });
    if (!r.ok) { showTplMsg(extractError(r.status, r.data), true); return; }
    showTplMsg(`已创建 ${id} v${r.data.version}，变量: ${r.data.variables.join(', ') || '无'}`, false);
    await refreshTemplates();
    await openTemplate(id);
  } else if (state.editor.mode === 'edit' && state.editor.viewingVersion === null) {
    const r = await api('POST', `/v1/prompts/${state.editor.tplId}/versions`, { content });
    if (!r.ok) { showTplMsg(extractError(r.status, r.data), true); return; }
    showTplMsg(`已发布 v${r.data.version}`, false);
    await refreshTemplates();
    await openTemplate(state.editor.tplId);
  }
}

async function deleteTemplate() {
  if (state.editor.mode !== 'edit') return;
  const id = state.editor.tplId;
  if (!confirm(`确定删除模板「${id}」及其全部版本历史？`)) return;
  const r = await api('DELETE', `/v1/prompts/${id}`);
  if (!r.ok) { showTplMsg(extractError(r.status, r.data), true); return; }
  resetEditor();
  await refreshTemplates();
  showTplMsg(`已删除 ${id}`, false);
}

async function switchVersion(val) {
  if (val === 'latest') {
    state.editor.viewingVersion = null;
    $('#tpl-content').value = state.editor.baselineContent;
    setVarDefs(state.editor.currentVars);
    updateVersionHint(true);
    $('#btn-tpl-save').disabled = false;
  } else {
    state.editor.viewingVersion = parseInt(val, 10);
    const r = await api('GET', `/v1/prompts/${state.editor.tplId}/versions/${val}`);
    if (r.ok) {
      $('#tpl-content').value = r.data.content;
      setVarDefs(r.data.variables);
    }
    updateVersionHint(false);
  }
  syncPromptRefControls();
}

// ==================== 6. 变量表单与渲染预览 ====================

function setVarDefs(defs) {
  // 重建表单时保留同名变量的已填值
  const saved = {};
  document.querySelectorAll('#variable-form input[data-var]')
    .forEach(inp => { saved[inp.dataset.var] = inp.value; });
  state.varDefs = defs;
  renderVariableForm(saved);
}

function renderVariableForm(saved = {}) {
  const box = $('#variable-form');
  box.textContent = '';
  for (const name of state.varDefs) {
    const row = el('div', 'var-row');
    row.append(el('label', null, name));
    const input = document.createElement('input');
    input.dataset.var = name;
    input.placeholder = '留空不传；可填 JSON（数字/布尔/数组/对象）';
    if (saved[name] !== undefined) input.value = saved[name];
    row.append(input);
    box.append(row);
  }
}

function collectVariables() {
  const vars = {};
  document.querySelectorAll('#variable-form input[data-var]').forEach(inp => {
    const { empty, value } = tryParseScalar(inp.value);
    if (!empty) vars[inp.dataset.var] = value;
  });
  return vars;
}

function syncPromptRefControls() {
  const use = $('#use-prompt');
  // 只有已存在（可引用 id）的模板才能被调用引用
  const canUse = state.editor.mode === 'edit' && !!state.editor.tplId;
  use.disabled = !canUse;
  if (!canUse) use.checked = false;
  const show = use.checked;
  $('#variable-form').hidden = !show || state.varDefs.length === 0;
  $('#btn-render-preview').hidden = !show;
  if (!show) $('#render-preview').hidden = true;
}

async function renderPreview() {
  if (state.editor.mode !== 'edit') return;
  const variables = collectVariables();
  const version = state.editor.viewingVersion ?? 'latest';
  const pre = $('#render-preview');
  const r = await api('POST', `/v1/prompts/${state.editor.tplId}/render`, { version, variables });
  const missing = (r.status === 400 && r.data && r.data.detail && r.data.detail.missing) || [];
  document.querySelectorAll('#variable-form input[data-var]').forEach(inp => {
    inp.classList.toggle('invalid', missing.includes(inp.dataset.var));
  });
  pre.hidden = false;
  if (r.ok) {
    pre.classList.remove('has-error');
    pre.textContent = r.data.rendered;
  } else {
    pre.classList.add('has-error');
    pre.textContent = `渲染失败: ${extractError(r.status, r.data)}`;
  }
}

// ==================== 7. 调用（流式/非流式） ====================

function buildPayload() {
  const payload = {
    model: $('#model-select').value,
    stream: $('#stream-toggle').checked,
    messages: [{ role: 'user', content: $('#user-message').value }],
  };
  const temp = $('#temperature').value.trim();
  if (temp !== '') payload.temperature = Number(temp);
  const mt = $('#max-tokens').value.trim();
  if (mt !== '') payload.max_tokens = parseInt(mt, 10);
  payload.thinking = $('#thinking-toggle').checked;
  // 结构化输出：schema 合法性在 validateBeforeSend 统一报错，这里解析失败就忽略
  if ($('#schema-toggle').checked) {
    try { payload.response_format = JSON.parse($('#schema-input').value); } catch { /* no-op */ }
  }
  if ($('#use-prompt').checked && state.editor.mode === 'edit') {
    payload.prompt = {
      id: state.editor.tplId,
      version: state.editor.viewingVersion ?? 'latest',
      variables: collectVariables(),
    };
  }
  return payload;
}

function setStreamState(s) {
  const dot = $('#stream-status');
  dot.className = 'dot ' + s;
  dot.title = STREAM_LABELS[s];
  $('#stream-status-text').textContent = STREAM_LABELS[s];
}

function setSending(on) {
  state.sending = on;
  $('#btn-send').disabled = on;
  $('#btn-abort').hidden = !on;
}

function resetResult() {
  const box = $('#result-text');
  box.textContent = '';
  box.classList.remove('has-error');
  $('#reasoning-box').hidden = true;
  $('#reasoning-text').textContent = '';
  const vb = $('#validation-box');
  vb.hidden = true;
  vb.textContent = '';
  vb.className = '';
  $('#result-meta').textContent = '';
}

// 结构化输出校验结果展示：通过显示绿条，失败保留流式原文并列出格式错误明细
function renderValidation(v) {
  const vb = $('#validation-box');
  vb.hidden = false;
  if (v.ok) {
    vb.className = 'pass';
    vb.textContent = '✓ JSON Schema 校验通过';
  } else {
    vb.className = 'fail';
    const lines = ['✗ JSON Schema 校验失败：'];
    for (const e of v.errors || []) {
      lines.push(`  [${e.field || 'root'}] ${e.type}: ${e.message}`);
    }
    vb.textContent = lines.join('\n');
  }
}

function renderResultError(text) {
  const box = $('#result-text');
  box.classList.add('has-error');
  box.textContent = text;
}

function renderMeta(m) {
  const parts = [];
  if (m.usage) parts.push(`in ${m.usage.input_tokens} tok / out ${m.usage.output_tokens} tok`);
  if (m.stop_reason) parts.push(`stop=${m.stop_reason}`);
  if (m.ttft_ms != null) parts.push(`ttft=${m.ttft_ms}ms`);
  if (m.elapsed_ms != null) parts.push(`elapsed=${m.elapsed_ms}ms`);
  $('#result-meta').textContent = parts.join(' · ');
}

function validateBeforeSend(payload) {
  if (!payload.model) return '请先选择模型';
  if (!payload.messages[0].content.trim()) return '请输入 user 消息';
  if ($('#schema-toggle').checked) {
    if (!$('#schema-input').value.trim()) return '已开启结构化输出，请填写 JSON Schema';
    try { JSON.parse($('#schema-input').value); } catch (e) {
      return `JSON Schema 不是合法 JSON: ${e.message}`;
    }
  }
  return null;
}

async function sendChat() {
  if (state.sending) return;
  const payload = buildPayload();
  const invalid = validateBeforeSend(payload);
  if (invalid) { resetResult(); renderResultError(invalid); setStreamState('error'); return; }

  resetResult();
  setSending(true);
  setStreamState('pending');
  state.abortCtrl = new AbortController();

  const entry = {
    id: `h_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    ts: new Date().toISOString(),
    model: payload.model,
    params: {
      temperature: payload.temperature ?? null,
      max_tokens: payload.max_tokens ?? null,
      stream: payload.stream,
      thinking: payload.thinking,
      schema: payload.response_format ?? null,
    },
    prompt: payload.prompt ?? null,
    message: payload.messages[0].content,
    result: null,
  };

  try {
    if (payload.stream) await sendStream(payload, entry);
    else await sendOnce(payload, entry);
  } catch (e) {
    if (e && e.name === 'AbortError') {
      // 保留已收到的部分文本
      if (!entry.result) entry.result = { text: '' };
      entry.result.error = '已手动取消';
      $('#result-meta').textContent = '已手动取消 · ' + $('#result-meta').textContent;
      setStreamState('idle');
    } else {
      renderResultError(`网络错误: ${e && e.message}`);
      entry.result = { error: `网络错误: ${e && e.message}` };
      setStreamState('error');
    }
  } finally {
    setSending(false);
    state.abortCtrl = null;
    if (entry.result) appendHistory(entry);
  }
}

async function sendOnce(payload, entry) {
  const r = await api('POST', '/v1/chat', payload, { signal: state.abortCtrl.signal, summarize: chatSummary });
  if (!r.ok) {
    const msg = extractError(r.status, r.data);
    renderResultError(msg);
    entry.result = { error: msg };
    setStreamState('error');
    return;
  }
  const d = r.data;
  // 结构化输出模式返回 structured_output 对象，普通模式返回 text
  const display = d.structured_output != null
    ? JSON.stringify(d.structured_output, null, 2)
    : (d.text ?? '');
  $('#result-text').textContent = display;
  renderMeta({ usage: d.usage, stop_reason: d.stop_reason, elapsed_ms: d.elapsed_ms });
  entry.result = {
    text: truncate(display, RESULT_SNIPPET),
    usage: d.usage ?? null,
    stop_reason: d.stop_reason ?? null,
    ttft_ms: null,
    elapsed_ms: d.elapsed_ms ?? null,
  };
  setStreamState('done');
}

async function sendStream(payload, entry) {
  const t0 = performance.now();
  let resultText = '';
  let reasoningText = '';
  let doneData = null;
  let errEvent = null;

  const r = await streamChat(payload, {
    signal: state.abortCtrl.signal,
    onEvent(event, data) {
      if (event === 'start') {
        setStreamState('streaming');
      } else if (event === 'delta') {
        if (data.channel === 'reasoning') {
          reasoningText += data.text;
          const box = $('#reasoning-box');
          box.hidden = false;
          box.open = true;
          $('#reasoning-text').textContent = reasoningText;
        } else {
          resultText += data.text;
          $('#result-text').textContent = resultText;
        }
        // 边收边更新历史条目：中途取消也能保留部分文本
        entry.result = { text: truncate(resultText, RESULT_SNIPPET), partial: true };
      } else if (event === 'done') {
        doneData = data;
      } else if (event === 'error') {
        errEvent = data;
      }
    },
  });

  const duration_ms = performance.now() - t0;
  addLog({
    method: 'POST',
    path: '/v1/chat (stream)',
    status: r.status,
    duration_ms,
    ok: r.ok && !errEvent,
    summary: doneData
      ? `in=${doneData.usage?.input_tokens ?? '?'} out=${doneData.usage?.output_tokens ?? '?'}`
        + ` · ${doneData.stop_reason ?? ''} · ttft=${doneData.ttft_ms ?? '?'}ms · elapsed=${doneData.elapsed_ms ?? '?'}ms`
      : '',
    request: payload,
    response: { text: truncate(resultText, SNIPPET), reasoning: truncate(reasoningText, SNIPPET), done: doneData, error: errEvent },
  });

  if (!r.ok) {
    const msg = extractError(r.status, r.data);
    renderResultError(msg);
    entry.result = { error: msg };
    setStreamState('error');
    return;
  }
  if (errEvent) {
    const msg = typeof errEvent.error === 'string' ? errEvent.error : (errEvent.error && errEvent.error.message) || JSON.stringify(errEvent.error);
    renderResultError(msg);
    entry.result = { error: msg };
    setStreamState('error');
    return;
  }
  renderMeta({
    usage: doneData?.usage,
    stop_reason: doneData?.stop_reason,
    ttft_ms: doneData?.ttft_ms,
    elapsed_ms: doneData?.elapsed_ms,
  });
  entry.result = {
    text: truncate(resultText, RESULT_SNIPPET),
    usage: doneData?.usage ?? null,
    stop_reason: doneData?.stop_reason ?? null,
    ttft_ms: doneData?.ttft_ms ?? null,
    elapsed_ms: doneData?.elapsed_ms ?? null,
    validation: doneData?.validation
      ? { ok: doneData.validation.ok, errors: doneData.validation.errors ?? null }
      : null,
  };
  // 流式结构化输出：done 携带校验结论。通过时用解析结果美化展示（与非流式一致），
  // 失败时保留流式原文并显示错误明细
  if (doneData?.validation) {
    const v = doneData.validation;
    if (v.ok && v.parsed != null) {
      $('#result-text').textContent = JSON.stringify(v.parsed, null, 2);
    }
    renderValidation(v);
  }
  setStreamState('done');
}

// ==================== 8. 日志 ====================

function addLog(entry) {
  state.seq += 1;
  entry.seq = state.seq;
  entry.time = fmtTime();
  state.logs.unshift(entry);
  if (state.logs.length > LOG_CAP) state.logs.length = LOG_CAP;
  renderLogs();
}

function renderLogs() {
  const list = $('#log-list');
  list.textContent = '';
  $('#logs-count').textContent = `${state.logs.length} 条`;
  for (const item of state.logs) {
    const det = el('details');
    const sum = el('summary');
    sum.append(
      el('span', 't', `[${item.time}] `),
      el('span', null, `${item.method} ${item.path} `),
      el('span', item.ok ? 's-ok' : 's-err', `→ ${item.status ?? 'ERR'} `),
      el('span', 't', `(${Math.round(item.duration_ms)}ms)`),
    );
    if (item.summary) sum.append(el('span', 't', ` · ${item.summary}`));
    det.append(sum);
    const dump = { request: item.request, response: item.response };
    if (item.error) dump.error = item.error;
    det.append(el('pre', null, JSON.stringify(dump, null, 2)));
    list.append(det);
  }
}

// ==================== 9. 历史与对比 ====================

function renderHistory() {
  const list = $('#history-list');
  list.textContent = '';
  if (!state.history.length) {
    list.append(el('li', 'empty', '暂无调用历史'));
    updateCompareBtn();
    return;
  }
  for (const h of state.history) {
    const li = el('li');
    li.dataset.id = h.id;
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = state.compareSel.includes(h.id);
    cb.addEventListener('change', () => toggleCompare(h.id, cb.checked));
    li.append(cb);
    li.append(el('span', 'h-time', fmtTime(new Date(h.ts))));
    li.append(el('span', 'h-model', h.model));
    li.append(el('span', 'h-prompt', h.prompt ? `${h.prompt.id}@${h.prompt.version}` : '无模板'));
    const isErr = !!(h.result && h.result.error);
    const stats = h.result && !h.result.error
      ? `out=${h.result.usage?.output_tokens ?? '?'} · ${Math.round(h.result.elapsed_ms ?? 0)}ms`
      : '失败';
    li.append(el('span', 'h-stats' + (isErr ? ' h-error' : ''), stats));
    const del = el('button', 'h-del', '✕');
    del.title = '删除该条';
    del.addEventListener('click', () => {
      state.history = state.history.filter(x => x.id !== h.id);
      state.compareSel = state.compareSel.filter(id => id !== h.id);
      saveHistory();
      renderHistory();
    });
    li.append(del);
    list.append(li);
  }
  updateCompareBtn();
}

function toggleCompare(id, checked) {
  if (checked) state.compareSel.push(id);
  else state.compareSel = state.compareSel.filter(x => x !== id);
  // 最多选 2 条：超出时挤掉最早的
  while (state.compareSel.length > 2) {
    const removed = state.compareSel.shift();
    const li = document.querySelector(`#history-list li[data-id="${removed}"]`);
    if (li) li.querySelector('input').checked = false;
  }
  updateCompareBtn();
}

function updateCompareBtn() {
  const btn = $('#btn-compare');
  btn.disabled = state.compareSel.length !== 2;
  btn.textContent = `对比所选（${state.compareSel.length}/2）`;
}

function openCompare() {
  const [a, b] = state.compareSel.map(id => state.history.find(h => h.id === id));
  if (!a || !b) return;
  const body = $('#compare-body');
  body.textContent = '';
  const aFaster = (a.result?.elapsed_ms ?? Infinity) <= (b.result?.elapsed_ms ?? Infinity);
  body.append(buildCompareCol(a, aFaster), buildCompareCol(b, !aFaster));
  $('#compare-dialog').showModal();
}

function buildCompareCol(h, isFaster) {
  const col = el('div', 'cmp-col');
  col.append(el('div', 'cmp-title', `${h.model} · ${fmtTime(new Date(h.ts))}`));
  const table = el('table');
  const rows = [
    ['temperature', h.params?.temperature ?? '默认'],
    ['max_tokens', h.params?.max_tokens ?? '默认'],
    ['stream', h.params?.stream ? '是' : '否'],
    ['深度思考', h.params?.thinking === undefined ? '默认' : (h.params.thinking ? '开' : '关')],
    ['结构化', h.params?.schema ? '是' : '否'],
    ['模板', h.prompt ? `${h.prompt.id}@${h.prompt.version}` : '无'],
    ['user 消息', truncate(h.message, 120)],
  ];
  if (h.result && !h.result.error) {
    rows.push(
      ['input tokens', h.result.usage?.input_tokens ?? '?'],
      ['output tokens', h.result.usage?.output_tokens ?? '?'],
      ['stop_reason', h.result.stop_reason ?? '?'],
      ['ttft', h.result.ttft_ms != null ? `${h.result.ttft_ms}ms` : '—'],
      ['elapsed', h.result.elapsed_ms != null ? `${h.result.elapsed_ms}ms` : '—'],
    );
    if (h.params?.schema) {
      rows.push(['结构化校验',
        h.result.validation ? (h.result.validation.ok ? '通过' : '失败') : '—']);
    }
  } else {
    rows.push(['结果', h.result?.error ?? '失败']);
  }
  for (const [k, v] of rows) {
    const tr = el('tr');
    const tdv = el('td', k === 'elapsed' && isFaster ? 'best' : '', String(v));
    tr.append(el('td', null, k), tdv);
    table.append(tr);
  }
  col.append(table);
  col.append(el('pre', null, h.result?.text ?? `（${h.result?.error ?? '无结果'}）`));
  return col;
}

// ==================== 10. init 与事件绑定 ====================

function bindEvents() {
  // 模板
  $('#tpl-search').addEventListener('input', renderTemplateList);
  $('#btn-tpl-new').addEventListener('click', editorNew);
  $('#btn-tpl-save').addEventListener('click', saveTemplate);
  $('#btn-tpl-delete').addEventListener('click', deleteTemplate);
  $('#tpl-version-select').addEventListener('change', (e) => switchVersion(e.target.value));
  $('#tpl-content').addEventListener('input', () => {
    if (state.editor.mode === 'new') {
      setVarDefs(draftVariables($('#tpl-content').value));
    } else if (state.editor.mode === 'edit' && state.editor.viewingVersion === null) {
      // 草稿期：服务端变量 ∪ 正则提取（值保留）
      const merged = [...state.editor.currentVars];
      for (const v of draftVariables($('#tpl-content').value)) {
        if (!merged.includes(v)) merged.push(v);
      }
      setVarDefs(merged);
      const changed = $('#tpl-content').value !== state.editor.baselineContent;
      $('#btn-tpl-save').disabled = !changed;
    }
    syncPromptRefControls();
  });

  // 调用
  $('#use-prompt').addEventListener('change', syncPromptRefControls);
  // 结构化输出与流式可同开：JSON 增量实时透传，done 事件携带 schema 校验结论
  $('#schema-toggle').addEventListener('change', (e) => {
    $('#schema-input').hidden = !e.target.checked;
  });
  $('#btn-render-preview').addEventListener('click', renderPreview);
  $('#btn-send').addEventListener('click', sendChat);
  $('#btn-abort').addEventListener('click', () => {
    if (state.abortCtrl) state.abortCtrl.abort();
  });

  // 日志
  $('#logs-bar').addEventListener('click', () => {
    const panel = $('#panel-logs');
    panel.classList.toggle('collapsed');
    $('#logs-toggle').textContent = panel.classList.contains('collapsed') ? '展开 ▲' : '收起 ▼';
  });
  $('#btn-clear-logs').addEventListener('click', () => {
    state.logs = [];
    renderLogs();
  });

  // 历史
  $('#btn-compare').addEventListener('click', openCompare);
  $('#btn-clear-history').addEventListener('click', () => {
    if (!state.history.length) return;
    if (!confirm('确定清空全部调用历史？')) return;
    state.history = [];
    state.compareSel = [];
    saveHistory();
    renderHistory();
  });
  $('#btn-close-compare').addEventListener('click', () => $('#compare-dialog').close());
  $('#compare-dialog').addEventListener('close', () => {
    state.compareSel = [];
    document.querySelectorAll('#history-list input[type="checkbox"]')
      .forEach(cb => { cb.checked = false; });
    updateCompareBtn();
  });
}

async function init() {
  loadHistory();
  renderHistory();
  resetEditor();
  bindEvents();

  const models = await api('GET', '/v1/models');
  await refreshTemplates();

  if (models && models.ok) {
    state.models = models.data.models;
    const sel = $('#model-select');
    sel.textContent = '';
    for (const m of state.models) {
      const opt = el('option', null, `${m.id}（${String(m.protocol).replace('Adapter', '')} 协议）`);
      opt.value = m.id;
      sel.append(opt);
    }
  }
}

init();
})();
