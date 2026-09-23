# PGSSI 私有权值配置

PGSSI 已注册为一等后端，但其预测需要**训练好的模型检查点**（`.pth`）。该检查点是研究组的私有产物，**绝不提交到本仓库**。本文档说明成员与部署方如何提供自己的权值，使功能可用，同时不把权值暴露给不应看到它们的人。

## 设计：权值保持私有，代码保持公开

```
公开仓库（git 跟踪）                       私有权值（git 忽略）
----------------------------------       ----------------------------------
thermo_engine/pgssi_backend.py            weights/PGSSI_best.pth   （或你机器上的
  读取 PGSSI_CHECKPOINT / PGSSI_SRC                              任意位置）
.env.example（说明各变量）                 .env                     （真实值）
docs/pgssi_checkpoint.md                  docker volume mount      （生产环境）
```

- 后端**只读取环境变量**；它本身既不包含路径，也不包含权值字节。
- `.env` 与 `weights/` 已在 `.gitignore` 中；`*.pth`/`*.pt`/`*.ckpt` 作为安全网也被全局忽略。
- 没有权值的成员只需把这些变量留空：PGSSI 保持注册状态，并返回结构化的 `missing_parameters` 失败，而**不会编造数值**。

## 必需变量

| 变量 | 含义 |
|---|---|
| `PGSSI_CHECKPOINT` | 训练好的 `*_best.pth`（或 `*_resume.pth`）绝对路径 |
| `PGSSI_SRC` | PGSSI 仓库 `src/models/PGSSI` 目录的绝对路径 |
| `PGSSI_HIDDEN_DIM` | 训练时使用的隐藏维度（默认 `512`） |
| `PGSSI_ENABLE_CROSS_INTERACTION` | 若启用了交叉交互则为 `1`（默认 `1`） |

## 本地开发

1. 将 `.env.example` 复制为 `.env`（此文件已被 git 忽略）。
2. 把上述四个变量设为**你自己的**私有路径。
3. 重启后端（`python -m uvicorn apps.api.main:app --port 8000`）。

这些值在请求时惰性读取，因此 `.env` 的加载顺序无关紧要。

## Docker 部署

`docker-compose.yml` 会透传 `PGSSI_*` 变量，并把宿主机的 `./weights` 以只读方式挂载到容器内的 `/weights`：

```bash
# 宿主机 .env
PGSSI_CHECKPOINT=/weights/PGSSI_best.pth
PGSSI_SRC=/opt/PGSSI/src/models/PGSSI
PGSSI_WEIGHTS_DIR=/absolute/host/path/to/weights
```

权值从宿主机挂载，**绝不烘入镜像**，因此镜像本身可以安全分享。

## 验证

配置好检查点后，请求无限稀释活度系数：

```
POST /api/calculations/infinite-dilution-activity
{ "components": [{"component_id":"ethanol","name":"ethanol","smiles":"CCO","cas_number":"64-17-5","aliases":[]},
                 {"component_id":"water","name":"water","smiles":"O","cas_number":"7732-18-5","aliases":[]}],
  "conditions": {"temperature_K": 298.15} }
```

预期结果：返回一个 `CalculationEnvelope`，其中 `model_name = "PGSSI"`，并含 `gamma_infinity` 数据点。若未配置检查点，同一请求会返回结构化的 `missing_parameters` 失败，并指明缺少检查点。
