# GitHub 团队协作设置

这份说明用于把当前本地工程安全地共享给团队。仓库中没有配置远程地址，也不会自动上传任何内容。

## 1. 首次创建远程仓库

在 GitHub 组织下创建一个空仓库，例如 `ThermoEqui-Agent`。建议先设为 Private，不要勾选自动生成
README、`.gitignore` 或 License，避免与本地文件冲突。

仓库管理员在本地执行：

```powershell
git branch -M main
git remote add origin https://github.com/<ORG>/ThermoEqui-Agent.git
git push -u origin main
```

如果使用 SSH，把远程地址换成：

```text
git@github.com:<ORG>/ThermoEqui-Agent.git
```

## 2. GitHub 仓库设置

在 `Settings → Collaborators and teams` 中按职责添加成员或团队：

- `maintainers`：仓库设置、发布和最终合并；
- `thermodynamics`：模型适用性、参数证据和验证评审；
- `backend`：Agent、API、数据库和确定性内核；
- `frontend`：Next.js 工作台与 API 契约。

在 `Settings → Branches` 为 `main` 建立 Ruleset：

- Require a pull request before merging；
- 至少 1 个 approval；科学模型与参数改动建议 2 个，其中 1 个来自热力学评审人；
- Require status checks：`backend`、`frontend`、`external-engines`；
- Require conversation resolution；
- Block force pushes 和 branch deletion；
- 仅维护者允许 bypass。

项目稳定后可以增加 `.github/CODEOWNERS`。团队名称确定前不要启用错误的自动指派。示例：

```text
/thermo_engine/                 @ORG/thermodynamics @ORG/backend
/knowledge/model_cards/         @ORG/thermodynamics
/knowledge/model_selection_rules/ @ORG/thermodynamics
/agent/                         @ORG/backend
/apps/api/                      @ORG/backend
/apps/web/                      @ORG/frontend
/schemas/                       @ORG/backend @ORG/frontend
```

## 3. 日常协作

每项工作都先建 Issue，再从最新 `main` 建短分支：

```powershell
git switch main
git pull --ff-only
git switch -c feat/123-nrtl-vle
```

完成后运行检查、提交并 Push：

```powershell
git add <明确的文件>
git commit -m "feat(engine): add NRTL VLE adapter"
git push -u origin feat/123-nrtl-vle
```

然后创建 Pull Request，关联 Issue，由另一名成员评审。不要在共享分支上使用 `git push --force`。
详细提交和测试要求见仓库根目录的 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 4. 推荐任务拆分

不要把“整合所有相平衡模型”放进一个 PR。更合适的拆分是：

1. 参数数据契约与证据字段；
2. 单个模型的后端适配器；
3. 模型适用性和排除规则；
4. 物理验证与基准案例；
5. API 契约；
6. 前端选择、状态和报告；
7. 文档与示例。

每个模型至少建立一个 Epic/父 Issue，再按“参数—计算—验证—界面”拆子任务。这样热力学人员可以审查
科学正确性，软件人员可以并行处理接口与 UI。

## 5. 密钥和数据安全

- 每位成员使用自己的 `.env`，只提交 `.env.example`。
- DeepSeek/OpenAI Key 只通过环境变量或 GitHub Actions Secrets 注入。
- GitHub Actions 的变量名可使用 `DEEPSEEK_API_KEY`，但默认 CI 不应调用付费外部 API。
- 运行数据库 `thermoequi.db`、日志、导出结果和测试缓存均不提交。
- 如果密钥曾出现在截图、终端分享或聊天记录中，应在供应商控制台立即轮换。
- 开启 GitHub Secret Scanning 和 Push Protection。

## 6. 发布建议

使用 `v0.1.0` 形式的 tag。发布前记录：

- 支持的模型、后端和计算类型；
- 参数库版本及来源；
- 验证基准和误差；
- 已知限制；
- 数据库迁移要求；
- 前后端镜像或安装方式。

若未来公开仓库，应先由团队选择并添加明确的开源 License；在此之前不要假定第三方可以复制或分发代码。
