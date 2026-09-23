# Contributing to ThermoEqui-Agent

本项目适合用 GitHub Issues + 短分支 + Pull Request 的方式协作。任何改动都必须守住两条边界：

1. LLM 只负责识别、规划、调用工具和解释，不能直接生成热力学计算结果。
2. 数值结果必须来自 `thermo_engine`，并通过 `validate_equilibrium_result`。

## 开发环境

建议使用 Python 3.11 或 3.12、Node.js 22 和 pnpm 11。

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -e ".[dev,phase-engines]"
pnpm --dir apps/web install --frozen-lockfile
Copy-Item .env.example .env
```

`.env` 仅保存在本机。不要提交 API Key、数据库文件、日志、构建目录或测试生成物。

## 领取任务

1. 在 GitHub 创建或领取一个 Issue，写清目标、范围和验收标准。
2. 一个分支只处理一个 Issue，推荐命名：
   - `feat/123-phasepy-vle`
   - `fix/234-flash-composition`
   - `docs/345-model-guide`
3. 开发前同步主分支：

```powershell
git switch main
git pull --ff-only
git switch -c feat/123-short-name
```

## 本地验证

后端与科学内核：

```powershell
python -m pytest
ruff check .
ruff format --check .
mypy .
```

前端：

```powershell
pnpm --dir apps/web test
pnpm --dir apps/web lint
pnpm --dir apps/web build
```

修改 Phasepy 或 Clapeyron 适配器时，还要运行：

```powershell
python -m pytest tests/test_external_backends.py
```

Clapeyron 完整集成测试需要 Julia，并设置 `RUN_CLAPEYRON_INTEGRATION=1`。

## 科学改动检查表

- 新模型实现了 `ThermodynamicBackend`，没有把第三方库对象暴露到 API。
- 缺失的二元参数返回结构化 `missing_parameters`，没有默认虚构参数。
- 参数包含来源、形式、方向、单位、适用范围和版本信息。
- 新的数值路径经过 `validate_equilibrium_result`。
- 至少增加一个公共后端或 HTTP 边界的行为测试。
- 测试夹具只在 `tests/fixtures`，生产代码没有导入测试数据。
- 超出 v0.1 范围的任务仍会被明确拒绝。

## 契约改动检查表

如果修改 `schemas/domain.py`：

1. 同步 `apps/web/src/lib/types.ts`。
2. 同步 API 请求/响应与 OpenAPI 测试。
3. 更新 `tests/test_frontend_contract.py`。
4. 如涉及持久化，同步 `database/models.py` 和仓储转换逻辑。

## 提交与 Pull Request

提交信息建议使用简洁的 Conventional Commits：

```text
feat(engine): add reviewed Wilson parameter contract
fix(agent): preserve explicit flash feed composition
docs: clarify Phasepy applicability
```

Push 后创建 Pull Request，并关联 Issue：

```powershell
git push -u origin feat/123-short-name
```

PR 应当保持可审查，说明科学假设、参数来源、测试证据和已知限制。至少一名其他成员批准且 CI 全绿后再合并。优先使用 squash merge，保持主分支历史清晰。

## 评审重点

评审顺序建议为：

1. 科学边界与参数证据；
2. 数值验证和失败行为；
3. API/前端契约一致性；
4. 可维护性、测试和文档；
5. UI 表达和工程体验。

详细目录说明见 [docs/repository-guide.zh-CN.md](docs/repository-guide.zh-CN.md)，GitHub 仓库设置见
[docs/github-collaboration.zh-CN.md](docs/github-collaboration.zh-CN.md)。
