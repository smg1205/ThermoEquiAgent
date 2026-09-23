# ThermoEqui-Agent 进度报告

> 更新日期：2026-08-27
> 范围：模型整合 + DWSIM 导出

本文记录 ThermoEqui-Agent 在**模型整合**和 **DWSIM 萃取精馏导出**两条线上的进度，
包括每样东西做完了没有、卡在哪、下一步做啥。

---

## 1. 模型整合进度

项目把机器学习模型接成**和普通模型一样的后端**，在
`thermo_engine/registry.py` 里注册，统一走同一套计算接口和校验流程。

### 1.1 现在有哪些模型

| 模型 | 类型 | 干什么用 | 状态 |
|------|------|---------|------|
| Ideal/Raoult | 普通计算 | 泡点、露点、闪蒸等 | 能用 |
| Peng-Robinson、SRK、RK、UNIFAC | 普通计算 | 相平衡 | 能用 |
| Phasepy/PR、Clapeyron/PR | 外部库 | 相平衡 | 能用 |
| NRTL、UNIQUAC、Wilson | 活度系数 | 相平衡、活度系数 | 部分能用 |
| **ThermoFormer** | 深度学习 | **预测 VLE（泡点、等压/等温）、活度系数** | 接好了，还没正式验证 |
| GHGEAT | 深度学习 | 预测物性 | 已注册，待核实 |

### 1.2 接入时守的规矩

- **不靠大模型算数**：深度学习模型只出预测值（比如 VLE），最后结果仍要过校验。
- **模型不冒充相平衡求解器**：模型只会它该做的预测，你去要它算它不会的，它会明确说
  "这个我不会/不在范围内"。
- **有适用范围限制**：
  - ThermoFormer：对组分数、压强、分子式（SMILES）都设了上限，超出会拒绝。
- **用到才加载**：模型权重、源码在执行时才读，不会因为缺 torch、rdkit 这些重库
  就拖慢整个项目的启动。

### 1.3 用到的环境变量（`.env`）

| 变量 | 作用 |
|------|------|
| `THERMOFORMER_SRC` | ThermoFormer 源码目录 |
| `THERMOFORMER_CHECKPOINT` | ThermoFormer 的权重文件 |
| `THERMOFORMER_USE_CUDA` | 是否用 GPU |

### 1.4 模型整合做到哪了

**做完了**
- ThermoFormer 后端接入、注册、接上校验了。
- 适用范围限制也加好了。

**还没做完 / 要确认**
- ThermoFormer 后端目前算**试验性质**，代码里明确写了"还没做基准测试、适用范围也还没正式审核"。
- 部署机器上可能没装 torch、rdkit 这类库，装了才能真正跑预测。

---

## 2. DWSIM 导出功能进度

### 2.1 能导出什么

`thermo_engine/dwsim_export.py` 里有两条导出口子：

| 函数 | 干什么 | 状态 |
|------|------|------|
| `export_dwsim_flowsheet` | 普通闪蒸流程 → `.dwxmz` 文件 | **能用**（验证过） |
| `export_dwsim_extractive_column` | 萃取精馏塔流程 → `.dwxmz` 文件 | 结构能生成，算的时候要手动补规格 |

### 2.2 萃取塔导出改过的几处

为了让萃取塔文件能正确生成，这几次修了这些问题（都有对应测试）：

1. **组分名规范化**
   - DWSIM 认组分名时**区分大小写**（写 `ethanol` 就报错，写 `Ethanol` 才行）。
   - 修法：把常见的小写名自动换成 DWSIM 认得的写法。

2. **改用它支持的塔类型**
   - 原来的简捷塔（ShortcutColumn）**只能接一条进料**，接不上"进料+萃取剂"两条，
     所以萃取剂连不上去。
   - 修法：改用严格塔（DistillationColumn），它能接多条进料。

3. **修正连接的端口**
   - 实测确认严格塔的端口：进料 `(0,0)`、萃取剂 `(0,1)`、塔顶 `(0,0)`、塔釜 `(1,0)`。
   - 修法：把萃取剂连到正确的端口；某条连接要是被 DWSIM 拒了，就记个提醒、
     不至于整个导出失败。

4. **适配塔参数的写法**
   - 严格塔的设置方法名和我们原来用的不一样（`set_NumberOfStages` 这类）。
   - 修法：多试几种写法，哪种能用用哪种。

### 2.3 现在的情况（重点）

- 生成的 `.dwxmz` 文件**能正常保存**，物料连接、塔板数、回流比都写进去了。
- 但严格塔在 DWSIM 里点**计算**前，还要手工补**冷凝器、再沸器的规格**和能量流。
  这几样在 DWSIM 的自动化接口里**没有稳定可靠的设置方法**，属于打开后在界面上配的。
- **设计本身是算好了的**（多少块板、回流比、塔顶塔釜温度都由我们自己的模型算出），
  只是落到 DWSIM 后，**第一次算要手动补一下规格**。

### 2.4 实测发现的 DWSIM 表现（留个底）

用诊断脚本（`scripts/diag_dwsim_extractive.py`）实际试出来的：

| 现象 | 结论 |
|------|------|
| DWSIM 认组分名区分大小写 | 要规范化（已修） |
| 简捷塔接不了第二条进料 | 改用严格塔（已修） |
| 严格塔进出端口各有 11 个 | 端口编号要看塔类型选 |
| 严格塔报 "stream connections missing" | 外部连的都接好了，但还是缺内部规格 |
| 严格塔用 MWK（泡点法）算了乙醇-水-乙二醇老不收敛 | 要改成 NS / Napthali（Newton 类） |
| 严格塔 Newton 报 "Error evaluating error functions" | 多半是**初始温度没给够**或**规格互相矛盾** |
| 有些 DWSIM 版本物性包下拉里没有 Wilson | 要靠初始值和自洽的规格来收敛 |

### 2.5 还没解决的麻烦和建议

- **想让萃取塔"下载下来就直接算出"在 DWSIM 自动化上是做不到的**：建能量流对象会
  报空引用，塔的内部规格也没有稳定的接口——这是 DWSIM 自动化本身的能力天花板，
  不是我们代码写错了。
- **现在的做法**：接受"文件能用 + 在 DWSIM 里手动补下规格"这种模式，具体步骤见
  `docs/extractive-dwsim-usage.zh-CN.md`。
- **如果想更省事**，以后可以：
  - 继续在代码里找建能量流的其他办法；
  - 或者直接给一份"设计报告 / JSON"，不依赖 DWSIM。

---

## 3. 代码 / 文档清单

| 文件 | 干什么的 |
|------|---------|
| `thermo_engine/dwsim_export.py` | DWSIM 导出（普通闪蒸 + 萃取塔） |
| `thermo_engine/column_design.py` | 萃取塔设计要求（Fenske/Underwood/Gilliland） |
| `thermo_engine/thermoformer_backend.py` | ThermoFormer 后端（预测 VLE） |
| `thermo_engine/registry.py` | 模型注册表 |
| `agent/extractive_distillation.py` | 认出萃取请求 + 安排导出 |
| `docs/extractive-dwsim-usage.zh-CN.md` | DWSIM 里手动配萃取塔的教程 |
| `scripts/diag_dwsim_extractive.py` | 实机排查脚本（开发用） |

---

## 4. 建议的下一步（按先后）

1. **把"萃取塔算不出来"的调法写进教程**：也就是"换 NS/Napthali 求解、补初始温度、
   精简规格"这几个操作，整理进 `extractive-dwsim-usage.zh-CN.md`。
2. **把深度学习后端从"试验"转成正式**：补基准测试、把适用范围审核掉，消掉
   "还没验证"的警告。
3. **（可选）加一个设计报告**：在不依赖 DWSIM 的情况下，把萃取设计结果输出成
   JSON 或 Markdown，作为兜底交付。
