# API

FastAPI 在 `/docs` 发布 OpenAPI。每个响应都带 `X-Request-ID`；错误格式为 `{"error":{"code","message","details","request_id"}}`。计算类路由会创建不可变的运行快照。

`GET /api/runs?limit=20&offset=0&status=passed|warning|failed` 按时间倒序返回轻量级运行摘要。`GET /api/runs/{run_id}/export?format=json|csv|dwsim` 返回可下载的产物。`dwsim` 格式通过本地 DWSIM Automation API 生成 `.dwxmz` 相平衡闪蒸流程。它依赖可选的 `dwsim` 依赖项，以及指向包含 `DWSIM.Automation.dll` 目录的 `DWSIM_HOME` 环境变量。

未预期的异常返回脱敏后的 `500 internal_server_error`，请求 ID 同时出现在 JSON 体和响应头中。**异常消息与数据库 / 提供方细节不会返回给客户端。**

会话编排器通过 `LLM_PROVIDER` 支持 `deterministic`、`deepseek` 和 `openai` 三种提供方。DeepSeek 使用其兼容 OpenAI 的 `/chat/completions` 端点；外部提供方可以结构化任务并加以解释，**但绝不产生热力学计算数值**。

主要路由：`/api/chat`、`/api/tasks/parse`、`/api/models/recommend`、`/api/models`、`/api/parameters`、`/api/parameters/search`、全部计算端点（含带类型的 LLE 契约 `/api/calculations/lle`）、`/api/validation`、运行查询与导出，以及 `/health`。

`POST /api/parameters` 在持久化前校验组分顺序、数值有限性、单位完整性、适用范围与来源证据。参数集 ID 不可变；提交已存在的 ID 返回 `409 duplicate_parameter_set`。测试夹具会被生产仓库拒绝。

聊天触发的运行与结构化的 `/api/calculations/*` 运行，都会在写入不可变运行快照之前先持久化归一化的任务清单。聊天任务保留其会话关联；直接计算任务的会话 ID 为空。

对于 `/api/chat`，会话、用户 / 助手消息、任务清单、运行、数据点、校验与证据记录**在同一个数据库事务中提交**。持久化失败会回滚整个响应快照，而不会留下一个没有计算结果的会话。
