# 三元液液平衡 LLE 内容摘要

资料来源：

- `manuscript-JACS-ja-2026-11644r.pdf`
- `manuscript-SI-JACS-ja-2026-11644r.pdf`

## 研究对象

论文提出 PSMI，完整名称为 Physics-informed site-specific molecular interaction learning framework，用于从分子结构重构三元液液平衡相图，并服务于萃取流程设计。

核心任务不是判断任意体系是否会分相，而是在已知目标体系存在 LLE 的前提下，预测三元体系的两相组成、binodal 边界和 tie-line 方向。

## 三元 LLE 输入与输出

模型输入：

- 三个组分的分子结构：`M1, M2, M3`
- 温度：`T`
- 相路径参数：`s`，范围为 `[0, 1]`

模型输出：

- 萃取相组成：`xE1, xE2, xE3`
- 萃余相组成：`xR1, xR2, xR3`

也就是每个预测点输出一条 tie-line 两端的相组成：

```text
extract phase:  [xE1, xE2, xE3]
raffinate phase:[xR1, xR2, xR3]
```

## 数据集

主要训练数据来自 Song 等人的离子液体三元 LLE 数据集。

筛选条件：

- 每个 system-temperature group 至少包含 6 条实验 tie-line
- 每条数据包含：组分名称、温度、萃取相摩尔分数、萃余相摩尔分数
- 分子结构使用 PubChem 和 OPSIN 获得 canonical SMILES

数据规模：

- 765 个三元体系
- 7,683 条实验 tie-line
- 数据按化学体系划分为训练、验证、测试，比例为 8:1:1

数据增强：

- 交换第 2、3 组分顺序
- 同步交换对应的相组成标签
- 用于增强模型对组分排列的鲁棒性

## 模型思想

PSMI 结合了数据驱动学习和热力学约束。

主要结构包括：

- 分子图表示
- 混合物图表示
- cross-attention，用于学习组分间相互作用
- Transformer，用于建模沿相路径 `s` 的变化
- 物理约束项，用于降低两相化学势不一致

模型目标是让预测的两相组成同时满足：

- 与实验 tie-line 接近
- 相组成非负
- 每相摩尔分数和接近 1
- 两相化学势差尽量小

## 主要性能指标

主模型在三元 LLE 测试集上的结果：

| 相 | MAE | RMSE | R2 |
|---|---:|---:|---:|
| 萃取相 | 0.0371 | 0.0566 | 0.9671 |
| 萃余相 | 0.0318 | 0.0545 | 0.9784 |

与多个基线模型相比，PSMI 的总体表现最好：

| 模型 | Overall MAE | Overall RMSE | Overall R2 |
|---|---:|---:|---:|
| PSMI | 0.0356 | 0.0700 | 0.9585 |

消融实验中，最佳结构为：

```text
Mixture graph + Cross-attention + S3 + Transformer
```

其总体结果为：

| MAE | RMSE | R2 |
|---:|---:|---:|
| 0.0330 | 0.0550 | 0.9742 |

## 物理约束效果

论文比较了纯数据驱动模型和 physics-informed 模型。

| 模型 | Overall MAE | Overall RMSE | Overall R2 | chemical-potential MAE |
|---|---:|---:|---:|---:|
| Data-driven | 0.0330 | 0.0550 | 0.9742 | 1.7711 |
| Physics-informed | 0.0349 | 0.0544 | 0.9748 | 0.5411 |

结论：

- 加入物理约束后，组成误差基本保持相近
- 两相化学势一致性显著改善
- 更适合用于后续流程设计，而不仅是拟合数据点

## 工业案例

### Case I: 芳烃萃取

体系：

```text
n-heptane + toluene + sulfolane
```

条件：

```text
T = 348.15 K
```

用途：

- 使用 sulfolane 从烷烃/芳烃混合物中萃取 toluene
- 对比 PSMI、实验数据、COSMO-RS、NRTL、UNIFAC
- 论文展示了三元相图和 tie-line 预测效果

### Case II: DEM 从水中回收

体系：

```text
water + diethoxymethane + p-xylene
```

条件：

```text
T = 303.15 K
```

用途：

- DEM 与水形成强非理想混合物
- 简单精馏不足以获得高纯 DEM
- 使用 p-xylene 作为萃取剂进行液液萃取
- 论文强调这是萃取/相平衡问题，不是二元 LLE 精馏问题

## 扩展数据与泛化实验

补充信息中还使用扩展文献 LLE 数据进行了微调。

扩展数据规模：

- 719 个三元体系
- 6,709 条 tie-line

划分：

- 575 个训练体系
- 72 个验证体系
- 72 个测试体系

训练设置：

- AdamW
- learning rate: `2e-5`
- weight decay: `1e-3`
- batch size: `256`
- max epochs: `200`
- gradient clipping: `1.0`
- mixed precision
- early stopping: 验证 RMSE 连续 15 个评估周期无改善

## 温度鲁棒性

论文对 6 个代表性体系做温度扰动：

```text
T + {-10, -5, 0, +5, +10} K
```

温度范围：

```text
283.00 K 到 353.20 K
```

结果：

- 平均组成温度敏感度：`5.82e-4 K^-1`
- 最大组分敏感度：`3.09e-3 K^-1`
- 沿 `s = 0..1` 的 101 个点预测时未出现负摩尔分数
- 最大相组成求和偏差：`1.19e-7`

## 系统级相图重构

测试对象：

- 78 个未见过的三元体系
- 803 条 tie-line

重构结果：

- 定量成功：4/78，约 5.1%
- 定性成功：72/78，约 92.3%
- 失败：2/78，约 2.6%
- 76/78，约 97.4%，保持了 binodal 趋势和 tie-line 方向

典型失败体系：

- `[C4MMIm][NTf2] / acetic acid / water`，293.15 K
- `[C4MIm][BF4] / 1-propanol / n-heptane`，298.15 K

## 与当前项目相关的理解

对 ThermoAgent/ThermoFormer 接入有用的结论：

- 论文重点是三元 LLE，不是 VLE，也不是精馏模型
- 输出应按两相组成理解，而不是泡点、露点或气液相组成
- 三元体系输入必须包含 3 个组分、温度和必要的组成/相路径信息
- 如果用于萃取流程设计，LLE 结果适合提供相分配、tie-line、萃取相/萃余相组成
- DWSIM 或流程模拟部分需要额外模块承接，论文模型本身不直接生成流程文件

