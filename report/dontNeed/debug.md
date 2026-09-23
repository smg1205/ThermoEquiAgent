# debug.md — 二元 VLE / DWSIM 导出链路：缺口与缺陷记录

> 记录时间：2026-09-16
> 范围：`agent/extractive_distillation.py`、`agent/orchestrator.py`、`schemas/column_design.py`、
> `apps/web/src/lib/types.ts`、`apps/web/src/components/Workbench.tsx`、
> `thermo_engine/dwsim_export.py`、`thermo_engine/column_design.py`
> 触发背景：把报告 §1.5-1 的「二元 VLE（2-丙醇/水）DWSIM 导出」接入 agent，随后排查遗留问题。
>
> 本文只记录**经实跑复现**的问题。每条都标注了复现命令与判定依据；未复现的猜测一律不写。

---

## 0. 本次改动引入的缺陷（最高优先级）

### BUG-1　`x_IPA=0.3` 这类单组分进料写法被判为「缺少进料组成」（**我的改动引入**）

**严重度：高**（会把报告里的标准写法直接挡在门外）

**现象**：报告通篇使用 `x_IPA=0.3` 记法。但该写法现在返回 `missing_parameters`：

```
'2-丙醇-水精馏塔，x_IPA=0.3，导出 dwsim'   → status=missing_parameters, missing=['feed_composition']
'isopropanol water distillation, x=0.3, dwsim' → status=missing_parameters
'甲醇-水精馏塔，甲醇0.3，导出dwsim'          → status=missing_parameters
'乙醇-水精馏塔，乙醇0.4水0.6，导出dwsim'      → status=dwsim_unavailable（这对能识别）
```

**根因**：我本次把 `_find_feed_mole_fractions()` 的兜底从 `[0.5, 0.5]` 改成了 `return None`。
该函数只接受**成对且和≈1** 的数值：

```python
for a in candidates:
    for b in candidates:
        if abs(a + b - 1.0) <= 1e-3:
            return [a, b]
return None          # ← 我改的这行
```

`x_IPA=0.3` 只出现**一个**数值，无法凑成和为 1 的数对，于是返回 `None`。
而 `run_binary_distillation()` 里有 `if fractions is None: return missing_parameters`。

**为什么之前没暴露**：旧代码无论解析成败都返回 `[0.5, 0.5]`，把这个问题**静默掩盖**了——
用户写 `x=0.3`，系统按 `x=0.5` 算，出的是错的塔板数却报 `ready`。所以旧行为是「错得看不出来」，
新行为是「对得用不了」。两者都不可接受。

**我的判断与建议**：`_find_feed_mole_fractions` 应当优先识别**带轻组分名的单值**写法，
再用 `1 - x` 补出重组分：

```python
# 建议：先抓 "x_IPA=0.3" / "甲醇0.3" / "异丙醇摩尔分数0.3" 这类单值，
#       按 message 顺序判断它属于 light 还是 heavy，另一组用 1-x 补全。
```

我**没有擅自实现**这一步，因为它需要确定「单值属于哪个组分」的判定规则（`x_IPA`、`x=`
无组分名、`甲醇0.3` 三种写法语义不同），改错会让**组成静默颠倒**——比现在的报错更危险。
建议明确规则后再改，并补一条 `x_IPA=0.3` ↔ `[0.3, 0.7]` 的回归测试。

**当前缓解**：用户写「异丙醇0.3 水0.7」或「乙醇40%水60%」这类成对写法可正常工作。

**已落地的回归测试**：`tests/test_binary_vle_dwsim.py` 新增 6 条针对性用例（**当前 6 条全红，正是本 BUG 的待修标记**）：

| 用例 | 覆盖 |
|---|---|
| `test_paired_feed_notation_is_parsed` | 成对写法（现状即可通过，防回归） |
| `test_absent_feed_is_reported_not_invented` | 无组成 → `missing_parameters`，不得默认 50/50 |
| `test_single_fraction_notation_implies_the_other_component` | `x_IPA=0.3` / `异丙醇摩尔分数0.3` / `甲醇0.3` / `x=0.3` → `[0.3, 0.7]`（4 条参数化） |
| `test_single_fraction_named_heavy_is_not_silently_inverted` | `水0.7` → `[0.3, 0.7]`（**防组成颠倒**） |
| `test_feed_parsing_does_not_swallow_temperature_or_pressure` | `80C` / `1 atm` 不得被当成摩尔分数 |

运行：

```powershell
python -m pytest tests/test_binary_vle_dwsim.py -p no:cacheprovider -v -k "single_fraction or does_not_swallow"
# 现状：6 failed
python -m pytest tests/test_binary_vle_dwsim.py -p no:cacheprovider -v -k "not single_fraction and not does_not_swallow"
# 现状：13 passed
```

其中 `test_single_fraction_named_heavy_is_not_silently_inverted` 是**最该保留**的一条：
它钉住了我在上文担心的「规则写错就静默颠倒塔顶塔底」这一风险。实现时若图省事写成
「第一个数字就是轻组分」，本条会立刻变红。

> 待 BUG-1 修完后，这 6 条应从「红」转「绿」，无需改动断言。

---

## 1. 先于我存在的缺陷

### BUG-2　DWSIM 未生成文件时仍返回 `file_id`（幽灵文件 ID）

**严重度：中**（前端若用它拼下载链接必然 404；当前前端靠 `status` 判断才没触发）

**现象**：导出失败、`dwsim_file_uri` 为 `None` 时，`file_id` 却是个真实格式的字符串，
但磁盘上**没有**这个文件。实跑复现：

```
LLE                  status=failed               uri=None   PHANTOM_file_id=False
extractive-ethanol   status=dwsim_unavailable    uri=None   PHANTOM_file_id=True
      file_id=extractive-20260916-212338-2ac331df
eac-npac-dmso        status=dwsim_unavailable    uri=None   PHANTOM_file_id=True
      file_id=eac-npac-dmso-vle-20260916-212339-9673017f
ipa-extractive       status=dwsim_unavailable    uri=None   PHANTOM_file_id=True
      file_id=ipa-extractive-20260916-212340-6c8df067
```

**根因**：两处赋值顺序问题。

1. `run_extractive_export()` / `run_eac_npac_dmso_vle_export()` / `run_ipa_extractive_export()`：
   `file_id` 在 `try` **之前**就已生成并赋给 `file_id` 变量，`except` 分支只清 `dwsim_uri`，
   **没有清 `file_id`**。
2. `run_lle_extraction_export()` 走的是提前 `return` 的写法，所以天然没有这个问题
   ——这也说明两种写法的行为不一致。

**验证**：用该 `file_id` 直接请求下载端点，得 404：

```
GET /api/export/extractive/ipa-extractive-20260916-212051-bb90056b.dwxmz
→ 404 {'error': {'code': 'http_404', 'message': 'Extractive DWSIM export not found', ...}}
```

**建议**：在这三个 `except` 分支里一并 `file_id = None`，与 LLE 路径及我新写的二元路径保持一致。

**备注（含归属证据限制）**：我本次新加的二元 VLE 路径从一开始就写了 `file_id = None`，
实测 `PHANTOM_file_id=False`，可作为修法参照。这三处旧路径的 `except` 分支在我改动前
即**只清 `dwsim_uri`、不清 `file_id`**（本次会话读取该文件时已如此）；但同 BUG-3，
`agent/` 未被 git 跟踪，无法给出 `git diff` 级证据。

### BUG-3　`_DISTILLATION_BINARY_ALIASES` 的 heavy 侧只认中文「水」

**严重度：中**（英文/中英混写请求被漏识别）

**现象**：`methanol water distillation column design dwsim export` 无法识别体系。

**根因**：表里 heavy 侧存的是**单个字符串** `"水"`，匹配逻辑是：

```python
has_heavy = heavy_text in lower or (heavy_text == "ˮ" and "water" in lower)
```

那个 `"ˮ"` 是个**不可见/畸形的字符**（U+02EE），与 `"水"` (U+6C34) 并不相等，
所以 `or` 右侧**永远为假**，`water` 永远匹配不上。

**已修**：我在本次改动中把该表改成 heavy **token 元组**并逐 token 匹配：

```python
(("甲醇", "methanol"), ("水", "water", "h2o"), ("methanol", "water")),
```

**关于归属的说明**：该 `"ˮ"` 畸形字符**先于我本次改动存在**（我读取该文件时它已在第 202 行）。
不过本仓库的 `agent/` 目录**未被 git 跟踪**（`git ls-files` 无此文件），因此无法用 `git diff`
给出提交级证据——判定依据是本次会话对文件的原始读取内容。

**副作用（重要）**：修好英文识别后，`"distillation column for methanol and water"`
从前因「体系识别失败」而落到 `missing_parameters`，现在能识别体系、继而暴露 BUG-1。
即 `tests/test_column_design.py::test_agent_binary_distillation_missing_feed` 那条断言
是**被 BUG-3 的遮蔽效应**养着的。修掉 BUG-3 后该用例才有真实意义。

---

## 2. 环境/工具链缺口（非代码缺陷，但阻塞验证）

以下三条都**不是**代码问题，但在当前 Windows + DSH 沙箱下会让人误判为 bug，必须记录。

### ENV-1　本机无法运行真实 DWSIM 导出

pythonnet 在本沙箱无法初始化：

```
RuntimeError: Failed to initialize Python.Runtime.dll
  → System.ComponentModel.Win32Exception: 拒绝访问。
  → System.Diagnostics.ProcessManager.OpenProcess(...)
```

**影响**：所有 `.dwxmz` 的真实写盘**从未被验证**。测试里 `export_dwsim_binary_column`
等一律是 mock。任何实跑都会落到 `dwsim_unavailable`。

**结论**：报告 §1.5-1 的设计数值我已逐个复现（见下），但**「文件能被 DWSIM 打开」这一条我无法证实**。
需在非受限终端（`conda activate thermo` 后）跑 `scripts/generate_ipa_water_std_column.py` 复核。

### ENV-2　`pytest` 默认 basetemp 在沙箱内不可用

`pyproject.toml` 配了 `--basetemp=.pytest-tmp`，但该目录在本沙箱被拒：

```
PermissionError: [WinError 5] 拒绝访问。: '\\?\E:\...\.pytest-tmp'
```

pytest 的 tmp 清理会 `chmod`，沙箱一律拒绝；换到 `$env:TEMP` 或工作区内的 `--basetemp`
**同样失败**。所以：
- 使用 `tmp_path` / `tmp_path_factory` 的用例在本机一律 ERROR（如 `test_dwsim_extractive_export.py`、
  `test_extractive_chat.py` 的部分用例）——**未改动的模块也一样失败**，可确认与本次改动无关。
- 我把新测试的导出目录改用工作区内 `mkdir` 创建，绕开该限制。

**注意**：CI 跑在 `ubuntu-latest`，**不受**此问题影响。

**附带的残留物**：排查过程中用 `tempfile.mkdtemp()` 在工作区根目录留下了若干 `tbv_*` 空目录
（8 个）。它们**是空的**（`write_bytes` 同样被拒，未写入任何文件），但沙箱不允许当前进程删除
由先前调用创建的目录，故未能清理。若在正常环境，直接 `Remove-Item -Recurse tbv_*` 即可。
这属于同一类沙箱权限问题，非代码缺陷。

### ENV-3　`ruff` / `mypy` / `pytest-asyncio` / `pnpm` 均不可用

- `ruff`、`mypy`：未安装，`pip install` 被沙箱/网络阻断 → **代码风格与类型门未跑**。
- `pytest-asyncio`：未安装 → 仓库既有的 `async def test_*`（如 `test_extractive_chat.py`）
  报 "async def functions are not natively supported"。**与本次改动无关**（这些用例早于本次改动）。
  我的新测试因此改用 `asyncio.run(...)` 包在同步用例里，规避该依赖。
- `pnpm` 不在 PATH，`npx` 因 PowerShell 执行策略被禁；直接用 `node` 跑 vitest 时
  esbuild 报 `spawn EPERM`（沙箱禁止 node 以管道方式 spawn 子进程）→ **前端测试未能执行**。
  前端改动（`types.ts` 加字段、`Workbench.tsx` 加分支）仅通过静态核对确认契约一致，
  **未经 vitest/tsc/build 验证**。

---

## 3. 已验证正常的部分（供对照，避免误判）

### 3.1 `report/dwsim/ipa_water_binary_column_x0p3_unifac.dwxmz` 实物比对

我把该 `.dwxmz` 解包（它是 ZIP，内含单个 `*.xml`）逐字段读出，与我们的导出器配置对比。
**结论：逐项吻合，无缺陷。**

真实文件结构：

```
物料流 : Feed / Distillate / Bottoms      （Tag 在 GraphicObject，Name 为 MAT-<uuid>）
塔     : "Binary Distillation Column"      Type = DistillationColumn
物性包 : UNIQUACPropertyPackage            Tag = "UNIQUAC"
组分   : Isopropanol, Water
进料组成: Isopropanol 0.3 / Water 0.7      （SpecType = Temperature_and_Pressure, StreamPhase = L）
进料级 : StreamBehavior=Feed -> AssociatedStage = a48f33ab-… = Stage8
板序列 : [0]=Condenser, [1..17]=Stage1..Stage17, [18]=Reboiler  （共 19）
```

逐项对照：

| 项目 | 真实 `.dwxmz` | 我们的导出器 | 判定 |
|---|---|---|---|
| 组分 | Isopropanol, Water | isopropanol, water | OK |
| 物性包 | **UNIQUAC** | `_add_property_package(flowsheet, "UNIQUAC")` | OK |
| 进料组成 x_IPA | 0.3 / 0.7 | 0.3 / 0.7 | OK |
| 理论板数 N | 19 | 19 | OK |
| 回流比 R | 2.694 | 2.694 | OK |
| **进料级** | Stage8 → 0 基索引 **8** | `feed_stage - 1` = **8** | **OK** |
| 冷凝器 | Total_Condenser | Total_Condenser（API 默认） | OK |
| 求解器 | Naphtali-Sandholm | `solving_method` 默认 Naphtali-Sandholm | OK |
| 最大迭代 | 500 | `max_iterations` 默认 500 | OK |
| 塔压 | 101325 Pa | 101.325 kPa → 101325 Pa | OK |
| 压降 | 5000 Pa | `pressure_drop_kPa=None` → 留 DWSIM 默认 | 注 |
| 板间距 | 0.5 m | 未写入 → 留默认 | 注 |

> **重要更正**：我最初用脚本比较时误报「进料级 DIFFER（9 vs 8）」。那是**我的比较脚本错了**——
> 拿引擎的 1 基编号 `feed_stage=9` 去比真实文件的 0 基索引 `8`，口径不同。
> 导出器实际写的是 `feed_stage - 1 = 8`，与真实文件**完全一致**。代码无此 bug。
> （`thermo_engine/dwsim_export.py:1140`：`_register_feed_on_stage(column_object, feed, feed_stage - 1, "feed")`）

**两处「注」**：压降与板间距我们未显式写入，真实文件里分别是 5000 Pa / 0.5 m。
这不影响可打开性（DWSIM 用自身默认值），但若要与归档文件**逐位一致**，需补写这两个字段。
`export_dwsim_binary_column` 已支持 `pressure_drop_kPa` 参数，`TraySpacing` 目前无对应入参。

### 3.2 其他已实跑确认项

以下均已实跑确认，不属缺陷：

| 项目 | 结果 |
|---|---|
| 二元 VLE + DWSIM 意图识别 | `2-丙醇-水…导出dwsim` / `isopropanol-water…download dwsim` / `methanol water…dwsim` → `True` |
| 负例不误触 | `甲醇-水精馏塔设计`（无 DWSIM 标记）→ `False`；`苯-甲苯 VLE 曲线，导出dwsim` → `False` |
| 萃取/LLE 优先级 | `乙醇-水萃取精馏，导出 dwsim`、`乙醇/乙酸乙酯/水液液萃取，导出 dwsim` → `False`（让位） |
| 路由 | 端到端 `chat()` → `intent=DISTILLATION_DESIGN`，`distillation.result` 非空 |
| 报告 §1.5-1 复现（x_IPA=0.3, UNIFAC） | α=2.713、N=19、N_min=10.069、R=2.694、R_min=1.924、进料板 9、355.26 / 367.46 K — 与实物 `.dwxmz` 逐项一致 |
| 报告 §1.5-1 复现（x_IPA=0.5, UNIFAC） | α=1.462、N=42、N_min=24.213、R=5.983、R_min=4.273、进料板 19 |
| 下载端点路径穿越防护 | `GET /api/export/extractive/..%2F..%2F.env` → 404（防护有效） |
| 前后端契约同步 | `BinaryDistillationPayload` 所有字段均在 `types.ts` 中声明；`feed_temperature_K`、`dwsim_unavailable` 均已同步 |
| 新测试 | `tests/test_binary_vle_dwsim.py` **11 passed**（另有 6 条 BUG-1 待修用例，见 §0） |
| 既有回归 | `test_column_design.py` 17 passed；`test_api_chat_blackbox.py` 8 passed；`test_schemas_validation.py` 10 passed；`test_frontend_contract.py` 1 passed |

---

## 4. 复现命令

```powershell
cd E:\codex\ThermoAgent\ThermoFormer\ThermoAgent

# BUG-1 复现
python -c "import sys; sys.path.insert(0,'.'); import agent.extractive_distillation as m; \
print(m.run_binary_distillation('2-丙醇-水精馏塔，x_IPA=0.3，导出 dwsim').status)"

# 新测试（同步用例，不依赖 pytest-asyncio）
python -m pytest tests/test_binary_vle_dwsim.py -p no:cacheprovider -v --tb=short

# 回归
python -m pytest tests/test_column_design.py -p no:cacheprovider -v --tb=no
```

> 注：本机跑 pytest 时会看到 `EXIT=-532462766`（0xE0434352，.NET 未处理异常），
> 这是 pythonnet 在进程退出阶段的崩溃，**发生在 pytest 打印 "N passed" 之后**，
> 不代表用例失败。判定请以输出中的 `N passed` 行与单模块 `exit=0` 为准。

---

## 5. 建议的修复顺序

1. **BUG-1**（高）：先定「单值进料属于哪个组分」的规则，再改 `_find_feed_mole_fractions`，
   并补 `x_IPA=0.3` 回归测试。**优先级最高**，因为它是报告标准写法的直接入口。
2. **BUG-2**（中）：三个 `except` 分支补 `file_id = None`；顺手统一 LLE 与其他路径的写法。
3. **ENV-1**：在非受限终端实跑一次 `scripts/generate_ipa_water_std_column.py`，
   用 DWSIM GUI 打开产出的 `.dwxmz` 确认桶流程与操作点，补齐「文件可用」这一未验证环节。
4. **ENV-3**：在能跑的环境补 `ruff check . && ruff format --check . && mypy .` 与前端
   `vitest / lint / build`，确认本次前端改动通过门禁。
5. **可选**：`is_extractive_distillation_request()` 对英文 `extractive distillation`
   仍要求「主动动词」才命中；纯名词短语会落到别的路由。若英文场景重要，建议一并收紧。

---
---

# 追加记录 A — 二元 LLE 萃取接入后的复查（2026-09-16，第二轮）

> 触发背景：把「二元 LLE 萃取 → 生成 DWSIM 文件」接入 agent（按 `report/dwsim` 二元 LLE 模板）。
> 本轮把上文 §2 的三条 ENV 缺口**重新实测**，其中 **ENV-1 已确认失效**，务必以本节为准。

## A.0 本轮新增交付（已验证可用，非缺陷）

> ⚠️ 本节在第二轮复查后**已更新**：最初交付版本存在 **BUG-8（保存了未求解的空壳文件）**，
> 已修复。详见 A.0.1。

| 验证项 | 结果 |
|---|---|
| 真实 DWSIM 生成二元 LLE `.dwxmz` | ✅ 20,250 字节，NRTL 物性包，**已求解** |
| 三方结构比对（官方脚本产出 / `report/dwsim` 模板 / 我的产出） | ✅ 结构指标**逐项完全相同**（见 A.0.1） |
| 相分率回读（Feed） | ✅ 与模板**逐位相同**：L1 frac=0.748107 x=[0.39915, 0.60085]，L2 frac=0.251893 x=[0.00553, 0.99447] |
| 二元 LLE ↔ 二元 VLE 路由互斥 | ✅ |
| 三元萃取优先级未被抢 | ✅ |
| 新增测试 `tests/test_binary_lle_dwsim.py` | ✅ **21 passed** |
| 回归 | ✅ 除既有失败外无新增（见 A.4） |

**顺带修掉的两个真实缺陷**：

- **1-Butanol 无 DWSIM 映射**：身份解析器返回 `1-Butanol`，但 DWSIM 字典键为**小写** `1-butanol`。
  实测 `1-Butanol` / `n-Butanol` / `Butanol` **均抛 `KeyNotFoundException`**。已补映射。
- **`25℃` 解析失败**：`_find_temperature` 中正则 `({n})\s*(?:℃|°c)\b` 的 `\b` 在 `℃` 后**永不匹配**
  （`℃` 非 ASCII 单词字符），导致 `25℃` 无法识别。已改为接受词边界或输入结尾。

---

### A.0.1 ⚠️ BUG-8（**本轮引入并已修复**）：保存了未求解的「空壳」流程文件

**严重度：高**（用户打开文件看到的是空流程，且**不报错**）

**发现方式**：用户质疑「仿照 `water_butanol_NRTL_default` 那个脚本写不就行了吗」，
于是把三个文件做逐项对照，暴露了差异。

**现象**：我最初交付的导出**结构正确但未求解**：

| 指标 | 我（修复前） | 模板 `NRTL_default` |
|---|---|---|
| `<Calculated>true</Calculated>` | **0** | 10 |
| `<Calculated>false</Calculated>` | **10** | 0 |
| `<Status>Calculated</Status>` | **0** | 5 |
| `<Status>NotCalculated</Status>` | **5** | 0 |
| XML 体积 | 172,174 | 206,834 |

即：文件能打开、拓扑对，但**每个单元都是 NotCalculated，相分率与组成全空**。

**根因**：我在 `export_dwsim_binary_lle_flowsheet()` 里**漏了 `CalculateFlowsheet4()` 调用**。
官方脚本 `scripts/generate_water_butanol_default.py`（第 108 行）与
`scripts/generate_water_butanol_dwsim.py` 都在 `_save_flowsheet_via_temp` **之前**先求解：

```python
automation.CalculateFlowsheet4(fs)      # ← 我漏了这一步
ded._save_flowsheet_via_temp(automation, fs, out)
```

**教训**：只按 DWSIM **对象模型**复刻拓扑是不够的，还必须复刻**求解步骤**。
结构对比（Tag、Vessel 类型）当时全部通过，正是因为我的比对**只查了结构、没查求解状态**。

**修复**：在保存前调用 `CalculateFlowsheet4`；求解失败**不丢弃文件**（结构仍有效，
用户可在 GUI 重跑），只打印 warning。

**修复后三方对照**（结构指标完全一致，仅字节数差几十，来自时间戳/GUID 长度）：

| 指标 | 官方脚本产出 | `report/dwsim` 模板 | 我的产出 |
|---|---|---|---|
| bytes | 20,239 | 20,237 | 20,250 |
| xml | 206,841 | 206,834 | 206,839 |
| `Calculated=true` | 10 | 10 | 10 |
| `Status=Calculated` | 5 | 5 | 5 |
| `AtEquilibrium=true` | 4 | 4 | 4 |
| objects / graphic | 5 / 5 | 5 / 5 | 5 / 5 |

**相分率回读**（`LoadFlowsheet` 后读 `Feed`，用模板脚本同款 API）——**逐位相同**：

```
Feed | z=[0.3000, 0.7000] | Liquid1: frac=0.748107 x=[0.39915, 0.60085]
                          | Liquid2: frac=0.251893 x=[0.00553, 0.99447]
Light_Liquid | z=[0.3992, 0.6008]   ← 有机相（富丁醇）
Heavy_Liquid | z=[0.0055, 0.9945]   ← 水相
```

**回归测试**（防再次退化成空壳）：

- `test_export_solves_the_flowsheet_before_saving` — 断言 `CalculateFlowsheet4` 被调用**恰好 1 次**
- `test_export_survives_a_failed_solve` — 求解抛异常时仍应产出可用文件

**复现（修复前）**：
```powershell
python -c "import zipfile,sys; sys.path.insert(0,'.'); from pathlib import Path; \
from agent.extractive_distillation import run_binary_lle_export as f; \
p=f('正丁醇-水二元液液萃取，0.3/0.7，导出 dwsim 文件', export_dir='.tmp/x'); \
z=zipfile.ZipFile(Path('.tmp/x')/(p.file_id+'.dwxmz')); x=z.read(z.namelist()[0]).decode(); \
print('Calculated=true:', x.count('<Calculated>true</Calculated>'), '(应为 10)')"
```

> **补充**：官方脚本里 `AddCompound` 失败后回退 `fs.AddCompound("N-butanol")` 是**死代码**——
> 实测 `N-butanol` / `n-butanol` / `1-Butanol` 在 DWSIM 9.0.5 中**均抛 `KeyNotFoundException`**，
> 只有小写 `1-butanol` 有效。我的映射表直接用 `1-butanol`，无需该回退。

---

## A.1 ⚠️ 对上文 ENV-1 的更正：本机**现在可以**跑真实 DWSIM

上文 ENV-1 称「pythonnet 在本沙箱无法初始化（拒绝访问），所有 `.dwxmz` 真实写盘从未被验证」。
**该结论已不成立**——本轮沙箱策略变更后，pythonnet 可正常初始化，我已**真实跑通**并产出文件：

```
DWSIM_HOME = C:\Users\34861\AppData\Local\DWSIM
REAL DWSIM OK: pythonnet initializes and CreateFlowsheet works
```

因此：

- 上文「§3 已验证正常」表中「文件能被 DWSIM 打开」这一**未验证项，现在可以补验**；
- **ENV-3 中「pythonnet 需要完整进程权限」的前提已变**，见 A.3 的剩余缺口。

**复现**：
```powershell
cd E:\codex\ThermoAgent\ThermoFormer\ThermoAgent
python -c "import os,sys; from dotenv import load_dotenv; load_dotenv(); sys.path.append(os.getenv('DWSIM_HOME')); import clr; clr.AddReference(os.path.join(os.getenv('DWSIM_HOME'),'DWSIM.Automation.dll')); from DWSIM.Automation import Automation3; Automation3().CreateFlowsheet(); print('OK')"
```

> **注意一个易踩的坑**：`import DWSIM` 会以 `ModuleNotFoundError: No module named 'DWSIM'` 失败。
> 必须先 `sys.path.append(DWSIM_HOME)` 再 `clr.AddReference(...)`。直接 `import` 不代表环境不可用。

---

## A.2 新发现缺陷（本轮，均**非**本轮改动引入）

### BUG-4　纯计算意图被误判为「导出文件」请求　🔴 高

**现象**：明确说"calculate"而非"export"的消息，被判为需要生成 DWSIM 文件。

```python
is_lle_extraction_request(
    "use ThermoFormer to calculate ethyl acetate n-propyl acetate DMSO LLE at 298.15 K and 101.325 kPa"
)  # -> True   （期望 False）
```

**根因**：`is_lle_extraction_request()` 只检查 ①LLE 关键词 ②三元体系别名表，
**完全没有检查交付物意图**（导出/dwsim/下载）：

```python
def is_lle_extraction_request(message: str) -> bool:
    lower = message.casefold()
    has_lle_marker = any(word in lower for word in ("lle", "liquid-liquid", ..., "液液萃取"))
    if not has_lle_marker:
        return False
    return _find_lle_system(message) is not None   # ← 从不问"要不要文件"
```

**影响**：用户说"帮我算一下乙醇/乙酸乙酯/水的 LLE 数据"，系统会**直接去生成 DWSIM 文件**，
而不是走相平衡计算。**用户可见的错误行为。**

**对照证据**：同文件的 `is_binary_vle_dwsim_request()` **正确**要求了 `wants_dwsim_file(message)`。
故这是 LLE 路由**漏了校验**，非设计如此。

**已有测试在报警**：`tests/test_lle_extraction_export.py:218`。
**本轮未触碰该函数**，与本次改动无关。

**建议修法**：末尾补 `return wants_dwsim_file(message) and _find_lle_system(message) is not None`，
并复核该文件中期望 `True` 的用例措辞是否都含导出标记。

---

### BUG-5　DWSIM 不可用时返回 `failed` 而非 `dwsim_unavailable`　🔴 高

**现象**：`run_lle_extraction_export()` 在异常下一律 `status="failed"`，
即使错误本质是"本机 DWSIM 未暴露萃取塔对象"这种**环境不可用**。

```python
except ThermoEquiError as exc:
    return LLEExportPayload(status="failed", ...)   # ← 应为 "dwsim_unavailable"
except Exception as exc:
    return LLEExportPayload(status="failed", ...)
```

**影响**：前端把 `failed` 当"设计算错了"，实际是"本机没配 DWSIM"。
用户无法区分「换环境重试」与「改参数」。对照 `BinaryDistillationPayload`、
`ExtractiveExportPayload` **都有** `dwsim_unavailable`——**只有 LLE 这条路径漏了**。

**并附一条硬事实**：DWSIM 9.0.5 的 Automation API **确实没有暴露液液萃取塔对象**，实测报错原文：

```
The installed DWSIM Automation API exposes no real liquid-liquid extraction tower object.
```

`_extraction_solver_member()` 探测的四个名字（`LiquidLiquidExtractor` /
`LiquidLiquidExtractionColumn` / `ExtractionColumn` / `LiquidLiquidColumn`）**全部不存在**。

> ✅ **这正是本轮二元 LLE 绕开的问题**：二元 LLE 用原生 `Vessel`
> （`report/dwsim` 模板的合法拓扑，不是"伪装的萃取塔"），**已实测可生成文件**。
> 结论：**二元 LLE 走 Vessel 可行；三元 LLE 想要真实萃取塔在本机 DWSIM 上做不到。**

**已有测试在报警**：`tests/test_lle_extraction_export.py:250, 267`。

---

### BUG-6　`_composition_argument` 返回类型随 `clr` 加载状态变化　🟡 中

**现象**：同一函数，pythonnet 已加载时返回 .NET `Double[]`，未加载时返回 Python `list`。

```python
type(_composition_argument([0.4, 0.6]))          # -> list,     [0.4, 0.6]
import clr
type(_composition_argument([0.4, 0.6]))          # -> Double[], System.Double[]
```

**根因**：`thermo_engine/dwsim_export.py` 用
`try: from System import Array, Double / except ImportError: return composition` 做兼容。
但一旦**进程内任何地方**加载过 pythonnet，该分支就变得可用，于是**同进程内后续调用返回类型改变**。

**影响**：造成**测试 order-dependent 失败**。典型证据：

```
E   assert <System.Doubl...002722DD65740> == [0.4, 0.6]
```

即 `tests/test_dwsim_export.py::test_export_creates_a_dwsim_phase_equilibrium_flowsheet`
**单独跑通过**，跟在 `tests/test_lle_extraction_export.py` 之后跑就**失败**。
这是「单测绿、全量红」的根因之一，**排查时极易误判成自己改坏了代码**
（本轮我就先怀疑了自己的改动，最后靠逐条对比失败清单才排除）。

**建议**：在**测试侧**用 `list(...)` 归一化断言，或 monkeypatch 该函数固定返回 list。
**不建议**改生产代码的 `.NET` 分支——那是喂给 DWSIM 的正确类型。

---

### BUG-7　三个既有导出路径在失败时仍返回「幽灵 `file_id`」　🟡 中

> 备注：上文 BUG-2 已记录此问题。本轮**独立复现确认仍然存在**，此处仅补充归属证据。

`run_extractive_export()` / `run_eac_npac_dmso_vle_export()` / `run_ipa_extractive_export()`
在 `dwsim_file_uri=None` 时仍返回一个**磁盘上并不存在**的 `file_id`：

```
extractive-ethanol   status=dwsim_unavailable  uri=None  PHANTOM_file_id=True
      file_id=extractive-20260916-212338-2ac331df
eac-npac-dmso        status=dwsim_unavailable  uri=None  PHANTOM_file_id=True
ipa-extractive       status=dwsim_unavailable  uri=None  PHANTOM_file_id=True
```

**根因**：`file_id` 在 `try` **之前**生成，`except` 分支只清 `dwsim_uri`、**不清 `file_id`**。
`run_lle_extraction_export()` 用提前 `return` 写法，天然无此问题——**两条路径行为不一致**。
本轮新增的二元 LLE 路径从设计起即 `file_id = None`，实测 `PHANTOM_file_id=False`，可作修法参照。

**归属证据限制**：`agent/` 未被 git 跟踪，无法给出 `git diff` 级证据。
判定依据是本次会话对文件的原始读取内容 + 独立复现。

---

## A.3 缺口（缺什么）

### GAP-1　`ThermoAgent/` 未被 git 跟踪　🔴 最高优先级工程缺口

```
git log --oneline -1        → 03a167b Create readme.md   （仅 1 个 commit）
git ls-files ThermoAgent/   → 0 行
```

**后果**：**无法用 `git diff` / `git stash` 回滚任何改动**，也无法区分"我改坏了"与"本来就坏"。
上文 BUG-3、BUG-7 都因此只能以"会话读取内容"作证据，**证据强度被削弱**。

本轮我是靠「临时禁用新代码 → 跑同一命令 → 对比失败清单逐条相同」才证明零回归的，**代价很高**。

**建议**：尽快 `git add ThermoAgent/` 建立基线 commit。**这是后续一切修复的安全前提。**

### GAP-2　`pytest-asyncio` 未安装 → 3 个模块**静默失去覆盖**　🟡 中

```
ERROR tests/test_deepseek_provider.py     - Failed: 'asyncio' not found in `markers` configuration option
ERROR tests/test_orchestrator_and_identity.py - Failed: 'asyncio' not found in `markers` configuration option
ERROR evals/test_agent.py                 - Failed: 'asyncio' not found in `markers` configuration option
```

因 `pyproject.toml` 设了 `--strict-markers`，未注册 marker 使**整个模块收集失败**
（是 error，不是 skip）——这 3 个模块里的测试**一个都没跑过**。

> 本轮我执行了 `pip install pytest-asyncio`（装了 1.4.0）以让套件可跑。
> **这是环境改动，不在原始需求内**；如需还原：`python -m pip uninstall -y pytest-asyncio`。

### GAP-3　测试依赖版本与 `pyproject.toml` 声明不一致　🟡 中

| 包 | `pyproject.toml` 钉的 | 实际安装 |
|---|---|---|
| pytest | 8.3.4 | **9.1.1** |
| pytest-asyncio | 0.25.2 | **1.4.0**（本轮新装） |

### GAP-4　`pytest.ini` 与 `pyproject.toml` 双份配置　🟢 低

- `ThermoFormer/pytest.ini` 存在但**内容为空（仅 `testpaths = tests`）**
- 真正的 `[tool.pytest.ini_options]` 在 `ThermoAgent/pyproject.toml`

两者都在会造成 rootdir 推导困惑。

### GAP-5　前端类型未同步 `binary_spec` / `lle_kind`　🟡 中（**待确认**）

`AGENTS.md` 要求「Schema changes require synchronized API models, frontend types, and contract tests」。
本轮在 `LLEExportPayload` 新增了 `binary_spec` 与 `lle_kind`，**尚未**同步
`apps/web/src/lib/types.ts`。需确认前端是否忽略未知字段（若用 zod strict 解析则会报错）。
**本轮未做前端验证**（`pnpm` 不可用，见 A.4）。

### GAP-6　API 层与前端未做端到端验证　🟡 中（**待确认**）

新特性仅在 orchestrator/agent 层验证。`/api/chat` 与下载端点
`/api/export/extractive/{filename}` **复用既有通路**，理论上无需改动，
但**未跑 API 层端到端测试**确认；前端 `lle_extraction` 渲染分支同样未验证。

---

## A.4 环境状态更新（对上文 §2 ENV 的复核）

| 缺口 | 上文结论 | 本轮复核 |
|---|---|---|
| ENV-1 pythonnet 无法初始化 | 无法跑真实 DWSIM | ❌ **已失效** — 真实 DWSIM 跑通（见 A.1） |
| ENV-2 `--basetemp=.pytest-tmp` 被拒 | tmp_path 用例一律 ERROR | ⚠️ **本轮未复现** — `.pytest-tmp` 存在且 `tmp_path` 用例可跑 |
| ENV-3 `ruff`/`mypy`/`pnpm` 不可用 | 门禁未跑 | ✅ **仍成立** — 三者均 not found |

**ENV-3 本轮实测**：

```
ruff : not found      （python -m ruff  → No module named ruff）
mypy : not found      （python -m mypy  → No module named mypy）
pnpm : not found
```

**遗留影响**：本轮改动的 `ruff check` / `ruff format --check` / `mypy` **仍未执行**；
前端 `vitest / lint / build` **仍未执行**。这些门禁需要在具备条件的环境中补跑。

**BNV 残留物**：上文 ENV-2 提到的 `tbv_*` 空目录本轮未再处理，仍在工作区根目录。

---

## A.5 全量测试套件现状（19 项既有失败）　🟡 中

**方法与免责声明**：本轮用「临时禁用新增代码 → 跑同一条命令 → 对比失败清单」验证，
**禁用前后失败清单逐条完全相同**，故下列 19 项**全部与本轮改动无关**。

| 失败测试 | 根因 |
|---|---|
| `test_lle_extraction_export.py` × 3 | BUG-4、BUG-5（真实缺陷） |
| `test_extractive_chat.py` × 3 | 依赖上述 LLE 行为 |
| `test_dwsim_export.py::test_export_creates_a_dwsim_phase_equilibrium_flowsheet` | BUG-6（order-dependent） |
| `test_dwsim_extractive_export.py` × 2 | `ModuleNotFoundError: No module named 'System'`（未先 `clr.AddReference`） |
| `test_model_catalog.py::test_real_model_catalog_yaml_files_all_load` | `assert 13 == 12` |
| `test_router_recommendations.py` × 2 | `ThermoFormer` 为"多余项"；`executable` 期望与实际不符 |
| `test_model_comparison.py` × 2 | `assert 2 == 3`；ThermoFormer 缺 SMILES 返回 `result=None` |
| `test_deepseek_provider.py` × 4 | 需真实 DeepSeek API |
| `evals/test_agent.py` + 2 模块 | GAP-2（`asyncio` marker 未注册） |

**其中值得单独决策的两类**：

1. **模型目录计数漂移**：`load_model_catalog()` 实际返回 **13** 个模型：
   `Clapeyron/Peng-Robinson, GHGEAT, Ideal/Raoult, NRTL, PGSSI, Peng-Robinson, Phasepy/Peng-Robinson, RK, SRK, ThermoFormer, UNIFAC, UNIQUAC, Wilson`。
   而测试硬编码期望 12、且不预期 `ThermoFormer`。**这是测试没跟上代码演进**，需有人裁定以哪边为准。

2. **ThermoFormer 缺 SMILES 时「标称可执行但静默失败」**：
   ```
   warnings=["ThermoFormer requires SMILES for component 'Benzene'."]
   executable=True 但 result=None
   ```
   即"声称可执行、实际执行不了"，前端可能显示成对比表里的空白项。
   按 `AGENTS.md`「不得编造数值」的精神，这类情况应**显式降级为不可执行**。

---

## A.6 建议修复优先级（合并两轮）

| 优先级 | 事项 | 理由 |
|---|---|---|
| P0 | **GAP-1** 建立 `ThermoAgent/` git 基线 | 没有它，后续修复都无法安全回滚 |
| P0 | **BUG-4** 纯计算意图误判为导出 | 用户可见错误行为，已有测试报警 |
| P0 | **BUG-5** `failed` vs `dwsim_unavailable` | 用户无法区分"换环境"与"改参数" |
| P1 | 上文 **BUG-1**（`x_IPA=0.3` 被判缺组成） | 报告标准写法的直接入口 |
| P1 | **GAP-2/GAP-3/GAP-4** 测试依赖与配置统一 | 3 个模块静默失去覆盖 |
| P1 | 裁定模型目录计数（A.5-1） | 需产品决策 |
| P2 | 上文 **BUG-2** / 本轮 **BUG-7** 幽灵 `file_id` | 前端若拼链接必然 404 |
| P2 | **BUG-6** 测试侧归一化 | 只影响测试稳定性 |
| P2 | **GAP-5/GAP-6** 同步前端类型、补 API 层测试 | 契约完整性 |
| P2 | ThermoFormer 缺 SMILES 应显式不可执行 | 避免"标称可执行但返回空" |

---

## A.7 本轮复现命令

```powershell
cd E:\codex\ThermoAgent\ThermoFormer\ThermoAgent

# BUG-4
python -c "import sys;sys.path.insert(0,'.');from agent.extractive_distillation import is_lle_extraction_request as f;print(f('use ThermoFormer to calculate ethyl acetate n-propyl acetate DMSO LLE at 298.15 K and 101.325 kPa'))"

# BUG-5
python -c "import sys;sys.path.insert(0,'.');from agent.extractive_distillation import run_lle_extraction_export as f;print(f('乙酸正丙酯和乙酸乙酯用DMSO萃取',export_dir='.tmp/xx').status)"

# BUG-6
python -c "import sys;sys.path.insert(0,'.');from thermo_engine.dwsim_export import _composition_argument as f;print(type(f([0.4,0.6])).__name__);import clr;print(type(f([0.4,0.6])).__name__)"

# 本轮新特性（真实 DWSIM，需 pytest-asyncio 已装）
python -m pytest -q tests/test_binary_lle_dwsim.py tests/test_binary_vle_dwsim.py tests/test_column_design.py

# 全量套件 + 既有失败清单
python -m pytest -q --continue-on-collection-errors 2>&1 | Select-String -Pattern "^(FAILED|ERROR)" | Sort-Object
```

## A.8 本轮变更文件索引

| 文件 | 变更 |
|---|---|
| `thermo_engine/dwsim_export.py` | 新增 `export_dwsim_binary_lle_flowsheet()`；补 1-butanol / MIBK / 氯仿 / 苯酚 / 乙醚 compound 映射 |
| `agent/extractive_distillation.py` | 新增 `is_binary_lle_dwsim_request()` / `run_binary_lle_export()` / `_binary_lle_components()`；修 `_find_temperature` 的 `℃` 解析 |
| `agent/orchestrator.py` | 二元 LLE 分类与分发到 `Intent.LLE_EXTRACTION` |
| `schemas/column_design.py` | `LLEExportPayload` 新增 `lle_kind`、`binary_spec` |
| `tests/test_binary_lle_dwsim.py` | **新增**，19 项测试 |
| `report/debug.md` | 追加记录 A（本节） |

---
---

# 追加记录 B — 按归档 `.dwxmz` 实物复刻二元 VLE（2026-09-16，第三轮）

> 触发背景：要求「按已有的 2-丙醇-水 DWSIM 文件」把二元 VLE 萃取流程写出来。
> 本轮**不再依据报告文字推断**，而是把归档文件解包逐字段读出，用**真实 DWSIM** 重新生成并对比。

## B.0 结论速览

| 项目 | 结果 |
|---|---|
| 真实 DWSIM 可用性 | ✅ 本机可跑（`DWSIM_HOME=C:\Users\34861\AppData\Local\DWSIM`） |
| agent 端到端真实生成 `.dwxmz` | ✅ `status=ready`，14,822 字节 |
| 与归档文件逐字段比对 | ✅ **15/15 全部一致**（含物性包、板数、回流比、进料级、求解器、压降、板间距） |
| 本轮修掉的缺陷 | **BUG-8**（压降漏写）→ 已修，比对由 12/13 变 15/15 |
| 阻塞项 | 🔴 **BUG-1 仍在**：`x_IPA=0.3` 走不到导出，见 B.3 |

## B.1 归档文件的真实结构（`report/dwsim/ipa_water_binary_column_x0p3_unifac.dwxmz`）

`.dwxmz` 是 **ZIP**，内含单个 `*.xml`（本例 227 KB）。关键字段：

```
物料流 : Feed / Distillate / Bottoms
         （Tag 在 <GraphicObject>，Name 为 MAT-<uuid>，两者分离，需按 ObjectID 关联）
塔     : "Binary Distillation Column"    Type = DWSIM.UnitOperations.UnitOperations.DistillationColumn
物性包 : UNIQUACPropertyPackage          Tag = "UNIQUAC"
组分   : Isopropanol, Water
进料   : Isopropanol 0.3 / Water 0.7     SpecType=Temperature_and_Pressure, StreamPhase=L
进料级 : <MaterialStreams> 内 StreamBehavior=Feed
         -> AssociatedStage = a48f33ab-… = **Stage8**
板序列 : [0]=Condenser, [1..17]=Stage1..Stage17, [18]=Reboiler （共 19）
```

> **读取要点**：进料级**不在** `<Stage>/<F>` 里（那里是空的），而在塔的
> `<MaterialStreams>` 块中由 `StreamBehavior="Feed"` + `AssociatedStage` 指定。

## B.2 缺陷 BUG-8　agent 路径漏写塔压降（**本轮修复**）

**现象**：agent 生成的二元塔，`ColumnPressureDrop` 为 `NaN`，而归档文件是 `5000` Pa。

**根因**：所有归档脚本都显式传了压降，agent 路径没传：

```
scripts/export_dual_source_columns.py:85    pressure_drop_kPa=5.0
scripts/generate_ipa_x0p3_column.py:116     pressure_drop_kPa=DP_KPA
scripts/generate_ipa_water_x0p5_column.py:73 pressure_drop_kPa=DP_KPA
agent/extractive_distillation.py            （缺失）
```

**影响**：文件仍可打开（DWSIM 用自身默认值），但与归档文件**不一致**，
且塔内压力分布与报告口径不同，严格复算时会对不上。

**修复**：`agent/extractive_distillation.py` 调用处补 `pressure_drop_kPa=_BINARY_DEFAULT_PRESSURE_DROP_KPA`，
常量取 `5.0`（= 归档的 5000 Pa），并注明来源。

**验证**：修复前后对比 —— 12/13 → **15/15**（新增压降、迭代容差两项亦一致）。

## B.3 🔴 BUG-1 仍未修，且它**直接阻塞**本轮工作

用真实 DWSIM 跑 agent 端到端，报告标准写法**走不到导出**：

```python
run_binary_distillation("2-丙醇-水二元VLE精馏塔设计，x_IPA=0.3，导出 dwsim")
# -> status = "missing_parameters"
# -> message = "未能解析进料组成（如甲醇50%、水50%），请补充。"
# -> file_id = None, dwsim_file_uri = None   ← 根本没生成文件
```

换成成对写法才通：

```python
run_binary_distillation("2-丙醇-水二元VLE精馏塔设计，异丙醇0.3 水0.7，导出 dwsim")
# -> status = "ready"
# -> file = binary-20260916-222702-1fc14e21.dwxmz (14,822 bytes)
```

**这印证了 §0 的判断**：BUG-1 不是「措辞挑剔」，而是**报告标准记法的硬阻塞**。
优先级维持最高。

## B.4 对 §0 / ENV-1 的更正

- **§0 的 BUG-2**（幽灵 `file_id`）：本轮复测，agent 的二元 VLE 路径仍为
  `file_id=None / uri=None`（一致，无幽灵）；三条**旧**路径（extractive / eac-npac-dmso /
  ipa-extractive）的幽灵 `file_id` 问题**依旧存在**，归并见记录 A 的 BUG-7。
- **ENV-1 已被记录 A 更正**：本机 pythonnet 可用。本轮据此完成了真实生成与比对，
  §3.1 中「文件能被 DWSIM 打开」这一**未验证项现已验证通过**。

## B.5 本轮验证命令

```powershell
cd E:\codex\ThermoAgent\ThermoFormer\ThermoAgent

# 1) 确认真实 DWSIM 可用
python -c "import os,sys;from dotenv import load_dotenv;load_dotenv();h=os.getenv('DWSIM_HOME');sys.path.append(h);import clr;clr.AddReference(os.path.join(h,'DWSIM.Automation.dll'));from DWSIM.Automation import Automation3;Automation3().CreateFlowsheet();print('REAL DWSIM OK')"

# 2) agent 端到端真实生成（注意用成对写法，x_IPA=0.3 会因 BUG-1 被拒）
python -c "import sys;sys.path.insert(0,'.');from dotenv import load_dotenv;load_dotenv();import agent.extractive_distillation as m;m.export_directory=lambda o=None:__import__('pathlib').Path('report/dwsim/_agent_regen');p=m.run_binary_distillation('2-丙醇-水二元VLE精馏塔设计，异丙醇0.3 水0.7，导出 dwsim');print(p.status,p.file_id)"

# 3) 解包归档文件核对结构
python -c "import zipfile;z=zipfile.ZipFile('report/dwsim/ipa_water_binary_column_x0p3_unifac.dwxmz');print(z.namelist())"
```

## B.6 本轮变更文件

| 文件 | 变更 |
|---|---|
| `agent/extractive_distillation.py` | 新增 `_BINARY_DEFAULT_PRESSURE_DROP_KPA = 5.0`；`export_dwsim_binary_column(...)` 补传 `pressure_drop_kPa=` |
| `report/debug.md` | 追加记录 B（本节）+ §3.1 实物比对表 |

> **未做**：`TraySpacing` 归档为 `0.5` m，我们新生文件恰好也是 `0.5`（DWSIM 默认值一致），
> 故无需补写；但 `export_dwsim_binary_column` **没有** `tray_spacing` 入参，
> 若将来需要显式控制间距，需新增参数。

---
---

# 追加记录 C — 二元 VLE 采出量与进料位置改造（2026-09-16，第四轮）

> 需求：二元 VLE 的 agent 导出，**塔釜流出量 = 入塔量的一半**，且 **feed 接在塔釜（上一级）**。

## C.0 本轮改动

`agent/extractive_distillation.py` 的 `run_binary_distillation()`：

| 项 | 改前 | 改后 |
|---|---|---|
| 塔釜采出 | 未写规格（`Product_Molar_Flow_Rate` 缺省 0） | **`bottoms = feed × 0.5` = 0.5 mol/s** |
| 塔顶采出 | 未写 | **`0.5 mol/s`**（= feed − bottoms） |
| 进料板 | `result.feed_stage`（短节法经验值 9） | **`stages − 1` = 18**（塔釜上一级） |
| 报文 | 报短节法原值 | 报**文件实际写入**的采出量与进料板 |

新增常量 `_BINARY_BOTTOMS_FEED_FRACTION = 0.5`。

**关键**：payload 的 `feed_stage` / `distillate_flow_mol_s` / `bottoms_flow_mol_s`
现在与写入文件的数值**同源**，避免"聊天里说的"和"文件里的"不一致。

## C.1 实测结果（真实 DWSIM 生成并解包核对）

```
status = ready
N stages = 19
N = 19   R = 2.694   dP = 5000
feed stage = Stage17 (DWSIM 0 基索引 17)
  spec C: Stream_Ratio              = 2.694
  spec R: Product_Molar_Flow_Rate   = 0.5
```

两项需求均落地：塔釜采出 **0.5 mol/s**（= 入料 1.0 的一半），进料接 **Stage17**
（0 基索引 17，即 19 级中的倒数第二级，紧邻 Reboiler）。

## C.2 ❗ 一个重要澄清：采出量此前**并非**为负

需求提到「塔釜流出量是负的」。实测**未复现负流量**：

```
                归档文件        本轮生成
Bottoms     M = +0.000000      +0.000000
Distillate  M = +0.000000      +0.000000
Feed        M = +1.000000      +1.000000
```

采出量是 **0**，不是负值。若在 DWSIM GUI 中看到负值，那是**塔内剖面**（内回流 L/V）
的显示——求解未收敛时的典型表现，见 C.3。

## C.3 🔴 新发现 BUG-9：严谨塔缺能量流连接，导致采出量恒为 0

**现象**：`auto.CalculateFlowsheet2(fs)` 返回成功，但 `Bottoms` / `Distillate`
的 `GetMolarFlow()` 始终为 **0**，即使 `Product_Molar_Flow_Rate = 0.5` 已正确写入。

**关键**：**归档文件同样如此**。这不是本轮引入的，也不是我改坏的：

```
===== ARCHIVED  ipa_water_binary_column_x0p3_unifac.dwxmz =====
CalculateFlowsheet2: OK -> {'Bottoms': 0.0, 'Distillate': 0.0, 'Feed': 1.0}
```

**直接原因**（DWSIM 自己报的）：

```
column.Calculate(): Exception: One or more of the stream connections to the
                    column is missing. Please check input data and try again.
                    在 DWSIM.UnitOperations.UnitOperations.Column.Validate
```

**根因**：塔的图形连接器 `in[10]` 是 **`ConEn`（能量输入）且 `attached=False`**：
冷凝器/再沸器的**能量流未连接**。归档文件与本轮生成文件**都是这个状态**。

```
in[0]  attached=True  type=ConIn     ← Feed
in[10] attached=False type=ConEn     ← 冷凝器/再沸器能量流（缺失）
out[0] attached=True  type=ConOut    ← Distillate
out[1] attached=True  type=ConOut    ← Bottoms
```

**影响**：这些 `.dwxmz` 目前只能"打开看结构"，**在 DWSIM 里按 Calculate 也不会算出结果**。
报告 §1.5 表 1.5-1 的设计数值来自确定性短节法，**不是**从这个文件求解出来的。

**建议修法**：导出时补两条 `EnergyStream` 分别接到塔的 `in[10]`/`out[10]`（冷凝器/再沸器），
再 `CalculateFlowsheet2`。这是让文件从"能打开"变成"能算出结果"的关键一步。
**本轮未修**，因为它超出"设采出量 / 改进料位置"这两项需求，且需要反射确认能量流接口。

## C.4 测试更新

`tests/test_binary_vle_dwsim.py`：

- `test_report_operating_point_reproduces_section_1_5_1`：`feed_stage` 断言由 `9` 改为 `18`
  （进料板随需求改变），并注明短节法设计值仍在核对范围内。
- **新增** `test_exported_product_split_is_half_of_the_feed`：断言塔釜 = 0.5、塔顶 = 0.5、
  两者之和 = 进料，且**导出器收到的参数**与 payload 一致。
- **新增** `test_exported_feed_tray_is_the_one_above_the_reboiler`：断言
  `feed_stage == stages - 1`，且导出器写入的 0 基索引为 17。

回归：**32 passed**（排除 §0 BUG-1 的 6 条待修用例）。

## C.5 本轮变更文件

| 文件 | 变更 |
|---|---|
| `agent/extractive_distillation.py` | 新增 `_BINARY_BOTTOMS_FEED_FRACTION`；采出量 = feed/2；进料板 = `stages-1`；payload 改报文件实际值 |
| `tests/test_binary_vle_dwsim.py` | 更新 1 条、新增 2 条测试 |
| `report/debug.md` | 追加记录 C（本节） |

> **遗留**：BUG-9（能量流缺失 → 采出量恒 0）与 BUG-1（`x_IPA=0.3` 被拒）均未修，
> 两者都不在本轮需求范围内，但都会影响"文件能不能真正算出结果"。
