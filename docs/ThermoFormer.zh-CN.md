# ThermoFormer：一个用于气液相平衡预测与自主分离设计的热力学感知分子相互作用框架

> 原文：`docs/ThermoFormer.pdf`（英文）。本文为其中文译文，依据 PDF 原文逐节翻译，保留公式与数据。

**作者**：Jinlin Ye, Yang Li, Minghao Sun, Hang Yuan, Shuohan Wang, Yu Wang, Lang Jiang, Xinchen Kang\*, Wei Zhang\*
（河北工业大学人工智能学院；河北工业大学化学工程学院；中国科学院大学化学与化学工程学院）

---

## 摘要

气液相平衡（VLE）是精馏、吸收、溶剂辅助分离等过程设计与优化的基础。准确的 VLE 数据对工艺开发至关重要，然而在快速扩展的化学空间中，对每种混合物都做完备的实验测量并不现实。以 NIST ThermoML 为代表的综合性数据库虽提供了宝贵的实验记录，但仅覆盖了可能混合物与热力学状态的一小部分。因此，需要通过预测性热力学模型，把这些稀疏的实验观测在组成、温度与化学体系三个维度上进行泛化。

经典活度系数模型通过物理动机明确的相互作用参数描述液相非理想性，包括 NRTL、Wilson、UNIQUAC 与 UNIFAC。NRTL、Wilson、UNIQUAC 一般需要针对具体体系进行参数回归，而 UNIFAC 通过预定义的基团与基团相互作用参数，把预测推广到未研究过的体系。不过，它们的适用性仍受限于可得的实验参数或已参数化的基团相互作用。量子化学方法如 COSMO-RS 与 COSMO-SAC 减少了对经验相互作用表的依赖，但引入了大量分子级计算且对计算设置敏感。

机器学习提供了另一条路径——直接从分子信息与热力学条件学习相平衡行为。早期人工神经网络（ANN）研究证明了数据驱动 VLE 预测在三元含盐体系与乙醇基二元混合物中的可行性。后续基于描述符的 ANN 与随机森林模型把 VLE 预测扩展到更宽的二元混合物空间，而 SMILES-RNN 架构引入分子序列与热力学性质用于联合露点与泡点预测。近年来基于图神经网络（GNN）的模型可以从分子图直接学习，减少了对人工设计分子描述符的依赖。预训练分子模型如 ChemBERTa-2、GROVER、Uni-Mol 提供了越来越可迁移的分子结构表示。对混合物，SolvGNN 明确引入了分子级相互作用学习，展示了二元与三元体系中依赖组成的活度系数预测。尽管有这些进展，多数直接 VLE 模型仍是二元取向，直接把分子表示映射为平衡可观测量，未能充分解析多组分相互作用效应。更重要的是，这种端到端映射未在学到的分子相互作用与由此产生的相行为之间提供显式热力学联系。这一局限促使热力学机器学习超越直接平衡性质回归，转而引入物理上有意义的中间变量。SPT-NRTL 把分子语言表示与 NRTL 框架结合，GDI-GNN 通过正则化引入 Gibbs–Duhem 一致性，GE-GNN 预测过量吉布斯能并通过自动微分得到活度系数，HANNA 把热力学一致性直接嵌入网络架构，TeNNet-SAC 则引入段活度系数作为物理中间变量，实现热力学一致的多组分预测。尽管如此，多组分可扩展性往往是通过投影或组合二元热力学信息实现，而非直接从多组分测量中学习。当额外组分会重组分子环境时，这种两两构造会受限——因为由此产生的多组分非理想性无法总能仅由组成二元行为恢复。

本文开发了一个用于多组分 VLE 预测与自主分离设计的热力学感知分子相互作用框架，其中 **ThermoFormer** 作为其核心分子热力学模型。ThermoFormer 用预训练分子模型编码每个组分，并把得到的分子 token 通过一个多组分相互作用 Transformer，使每个组分能根据周围物种的身份与组成更新其表示。热力学条件进一步把这些混合物感知的表示调制成组分特异的潜在非理想性状态，再由活度系数解码器转换为物理上有意义的活度系数，并随后用于可微 VLE 求解器以重建平衡温度、压力与汽相组成。模型训练受组成守恒、Gibbs–Duhem 一致性、纯组分极限、相图连续性与组分置换一致性约束，同时可微平衡求解器在 VLE 重建时施加相应的相平衡关系。最后，ThermoFormer 作为一个热力学 Skill 部署在 LLM 驱动的 Agent 中，协调相行为分析、溶剂筛选与过程模拟，用于自主分离设计。

---

## 2 结果

### 2.1 数据集构建与覆盖

从 NIST ThermoML 档案用 `thermoml-io` 提取实验 VLE 记录，标准化为二元与三元数据集。保留含分子身份、温度、压力、液相与汽相组成的记录，连同来源出处与可得的热力学一致性标注。温度与压力分别转换为 K 与 kPa，并根据组成闭合约束重建相关摩尔分数。用 RDKit 对分子结构进行规范化，用无序组分集定义体系身份。用优先级 SMARTS 规则划分化学族，未解析结构单独归为一类。

所得数据集包含 **28,290 个实验 VLE 点**，其中 **23,061 个二元观测来自 700 个体系**，**5,229 个三元观测来自 126 个体系**。二元与三元子集分别涉及 333 与 125 个分子组分，覆盖广阔且非均匀的温度–压力空间。化学覆盖包含 79 个无序二元体系对与 49 个三元体系组，结合了密集采样的类别与大量低频出现的化学类型长尾。组成空间从近纯组分极限延伸到混合物域内部。在三元体系中，30 个含全部三个组成二元子体系，22 个含两个，69 个含一个，5 个不含。这些分布支持在热力学状态内插、化学空间泛化与二元到三元迁移三方面进行受控评估。

### 2.2 预测性能与泛化

ThermoFormer 在整体预测、热力学状态泛化、未见组分外推与二元到三元迁移四方面进行评估。对每个随机种子，最终 checkpoint 只按验证集从监督与逸度微调候选中选出。（Table 1 汇总耦合的等温 P–x–y 与等压 T–x–y 结果。）

**2.2.1 整体预测性能** —— 仅二元训练得到的压力与温度 MAE 分别为 8.98±5.20 kPa 与 2.45±0.15 K，两个预测方向的汽相组成 MAE 均低于 0.03。联合二元–三元训练保持相当的二元性能（压力与温度 MAE 8.68±4.71 kPa 与 2.45±0.31 K）。联合训练模型预测三元压力与温度的 MAE 为 1.85±0.27 kPa 与 1.41±0.61 K。当联合评估二元与三元测试记录时，联合训练模型达到压力与温度 MAE 7.75±3.55 kPa 与 2.38±0.33 K，等温与等压任务的汽相组成 MAE 分别为 0.0251±0.0056 与 0.0299±0.0060。

**2.2.2 热力学状态泛化** —— 组成内插是最精确的情形（压力与温度 MAE $1.61\pm0.20$ kPa 与 $0.58\pm0.04$ K）。组成边缘外推保持压力与温度 MAE 分别在 2 kPa 与 1 K 以下。高温与高压外推使等温压力 MAE 分别增至 $5.83\pm0.42$ 与 $5.96\pm1.35$ kPa。汽相组成 $R^2$ 在高温与高压外推下仍保持 $\geq 0.995$。

**2.2.3 对未见组分的泛化** —— 未见组分在所有评估协议中产生最大的性能退化：压力 MAE 增至 $32.39\pm11.31$ kPa，温度 MAE 增至 $36.45\pm1.49$ K，对应 $R^2$ 分别为 $0.494\pm0.242$ 与 $0.456\pm0.044$。汽相组成 MAE 在等温与等压任务中达 $0.1057\pm0.0179$ 与 $0.1191\pm0.0062$。超出训练所代表化学空间的分子外推仍是当前模型的主要局限。

**2.2.4 二元到三元迁移** —— 无三元监督时，ThermoFormer 预测三元压力与温度的 MAE 为 1.62±0.38 kPa 与 1.91±0.89 K。零样本等温汽相组成 MAE 为 0.0129±0.0034，等压值为 0.0714±0.0167。提高三元训练比例产生非单调的性能轨迹：等温压力 MAE 最低出现在 50% 三元监督处，达 1.30±0.50 kPa。等压汽相组成误差在所有三元训练比例下保持窄范围。这些结果确立了直接的二元到三元迁移，而额外三元监督带来任务相关的增益。

> **Table 1（要点）**：等温压力 MAE（整体 $7.75\pm3.55$ kPa；组成内插 $1.61\pm0.20$；未见组分 $32.39\pm11.31$），等压温度 MAE（整体 $2.38\pm0.33$ K；内插 $0.58\pm0.04$；未见组分 $36.45\pm1.49$）。各单元格列报 MAE、RMSE、$R^2$。

### 2.3 与机器学习及热力学模型的比较

与六个二元与六个三元平衡曲线上的 NRTL、Wilson、UNIQUAC 比较。ThermoFormer 预测来自验证选定的 seed-0 checkpoint，不含体系特定重拟合。经典模型用 Phasepy 对每个显示体系的所有可得测量做单独拟合，蒸气压采用经验证的 DIPPR/Antoine/Wagner 关联。因此该比较对比的是体系特定追忆式拟合与体系无关分子预测。

**二元**：ThermoFormer 在四个中等案例上压力 MAE 1.17–1.92 kPa、温度 0.57–1.52 K、汽相组成 0.007–0.012。对 1-丙醇–乙醇，ThermoFormer 捕捉曲线趋势但压力 MAE 达 18.18 kPa，而拟合的经典模型约 2.42–2.45 kPa。最大二元偏差出现在异丁烷–丙腈，ThermoFormer 压力 MAE 达 50.38 kPa，经典模型误差 6.80–8.00 kPa；其汽相组成 MAE 仅 0.0015，表明 ThermoFormer 保留了液气组成映射但存在压力尺度偏移。

**三元**：低误差案例压力 MAE 1.19–1.52 kPa 或温度 0.67–1.00 K，汽相组成 0.005–0.016。代表性甲基环己烷–环己基胺–苯胺压力与汽相组成 MAE 1.24 kPa 与 0.028。21 个独特分子组分中 18 个有经验证的蒸气压关联，使 10 个独特体系中的 7 个可做经典计算。

在 8 个覆盖完整经典模型的案例中，NRTL/Wilson/UNIQUAC 经直接拟合后得到更低的状态变量 MAE。ThermoFormer 在 2/1/2 个案例中对汽相组成取得更低 MAE。经典模型在可得实验记录与经验证纯组分关联时提供高精度体系内表示；ThermoFormer 则提供体系无关预测，无需逐体系相互作用参数回归，并在外部纯性质输入不完整时保持覆盖。

### 2.4 消融分析

**2.4.1 分子表示** —— 显式 RDKit 描述子提供主要表示增益，相对单独 Uni-Mol v2 使压力与温度 MAE 分别降低 50.2% 与 52.0%。官能团特征单用不足以支撑稳定 VLE 重建（压力误差严重、求解器覆盖下降）。在 RDKit 描述子上叠加 Uni-Mol v2 无一致改进。加入官能团特征使压力 MAE 从 9.297±4.681 降至 8.205±3.309 kPa、温度从 2.578±0.450 降至 2.513±0.371 K。因此保留完整三视图表示作为最平衡的分子编码。

**2.4.2 多组分相互作用架构** —— 引入化学注意偏置与上下文条件对势不能一致优于香草多组分 Transformer。去掉注意偏置仍保留上下文条件对势会改善汽相组成与等压温度 MAE，但使压力 MAE 从 8.205±3.309 增至 8.741±4.253 kPa。附加相互作用机制在预测目标间重新分配精度而非产生一致增益。香草多组分 Transformer 因更强的压力预测、相当的温组成精度与更简单形式而被保留。

**2.4.3 逸度约束微调** —— 逸度约束微调改善 12 项平均预测指标中的 10 项，把教师强制逸度残差从 0.009237±0.003908 降至 0.008898±0.003440。唯一的总回归是温度 RMSE 小幅上升（4.146→4.202 K）。验证选择保留三个种子的微调 checkpoint、两个种子回到监督 checkpoint，移除总温度回归并改善全部 12 项平均指标。

### 2.5 学习到的多组分热力学可解释性

**2.5.1 分子视图依赖与跨视图相互作用** —— 分组 Shapley 归因显示一致的信息层级：RDKit 描述子占过量吉布斯能、活度系数、相对挥发度与对势强度的 59–65%，是主导的分子表示；官能团特征再贡献 23–33%，其对相对挥发度的贡献最大；Uni-Mol v2 直接贡献 8–13%。三视图作用各不同：全局物理化学描述子提供主信号，局部化学基序补充信息。特征遮蔽揭示对相对挥发度最有影响的 RDKit 描述子为 NH/OH 数、Balaban 连接度、极性表面积、logP、部分电荷与摩尔折射率；醛、酮、卤代、醇与酯基序主导官能团归因。

**2.5.2 依赖组成的分子相互作用** —— 对解析分析中，醇/多元醇–烷烃/环烷烃对贡献最大，其次为醇/多元醇–醚/羰基。多数主导对类呈正平均带符号贡献。显式留出的混合物显示模型能区分弱与强非理想二元体系：N,N-二甲基甲酰胺–二甲基亚砜接近理想；乙醇–异辛烷表现出更强且依赖组成的活度系数偏差。三元苯胺–甲基环己烷–环己基胺案例显示对贡献随第三组分占比增加而重新分布，说明学习到的三元非理想性反映依赖组成的分子相互作用重组，而非组成二元响应的固定叠加。

**2.5.3 逸度约束微调的作用** —— 微调降低全部五个随机种子的教师强制逸度 RMSE。改善集中在热力学中间变量而非最终汽相组成；饱和蒸气压预测变化最大，其次为活度系数，而汽相组成变化显著更小。逸度正则化主要通过重平衡纯组分挥发性与液相非理想性起作用，同时保持可观测平衡组成。

### 2.6 ThermoFormer 赋能的自主分离设计

（原文该节正文尚未展开；见配套实验文档 `ThermoFormer萃取精馏过程.md`。）

---

## 3 讨论

（原文该节未展开，正文留待补充。）

---

## 4 方法

### 4.1 ThermoFormer 公式化

ThermoFormer 针对二元与三元 VLE 体系的泡点预测而构造。对 $N$ 组分混合物（$N\in\{2,3\}$），分子结构记为 $\{M_i\}$，液相组成 $\mathbf{x}=(x_1,\dots,x_N)$，满足 $\sum_{i} x_i=1$。两种互补预测模式：

$$
\text{等温 P–x–y}: \quad (\{M_i\}, T, \mathbf{x}) \rightarrow (P_b, \mathbf{y})
$$

$$
\text{等压 T–x–y}: \quad (\{M_i\}, P, \mathbf{x}) \rightarrow (T_b, \mathbf{y}) \qquad \text{(式 1)}
$$

其中 $P_b$、$T_b$ 分别为泡点压力与泡点温度，$\mathbf{y}=(y_1,\dots,y_N)$ 为平衡汽相组成。

### 4.2 多视图分子表示学习

每个组分用三种互补分子视图表示：RDKit 描述子（$d_i^R$）、Uni-Mol v2 嵌入（$d_i^U$）、SMARTS 推导的官能团特征（$d_i^G$）。RDKit 视图编码显式物化与拓扑属性（分子尺寸、极性、亲脂性、氢键能力、环结构、原子组成、分子柔性），描述符值用相应训练划分上拟合的统计来标准化。Uni-Mol v2 视图提供从分子三维构象推出的高阶结构表示，用冻结编码器预计算并在 VLE 模型训练期间保持不变。官能团视图记录 SMARTS 基序的存在或出现次数（羟基、羰基、羧基、酯、醚、胺、酰胺、芳香、含硫、含卤素基团）。三种视图经独立投影网络：

$$
h_i^R = f^R(d_i^R), \qquad h_i^U = f^U(d_i^U), \qquad h_i^G = f^G(d_i^G) \qquad \text{(式 2)}
$$

投影视图拼接并变换为统一分子 token：

$$
h_i^0 = f^{\mathrm{fuse}}(h_i^R \,\|\, h_i^U \,\|\, h_i^G) \qquad \text{(式 3)}
$$

投影与融合网络在所有组分位置共享。$N$ 组分混合物的分子 token 收集为 $H^0=[h_1^0,\dots,h_N^0]$（式 4）。

### 4.3 置换等变多组分相互作用编码

融合分子 token 独立描述组成分子，但未考虑其混合依赖环境。向组分序列追加可学习混合物 token $h_{\mathrm{mix}}^0$：$H^{\mathrm{in}}=[h_{\mathrm{mix}}^0, h_1^0,\dots,h_N^0]$（式 5）。完整序列由共享多组分 Transformer 编码器处理：

$$
[h_{\mathrm{mix}}, \bar{h}_1, \dots, \bar{h}_N] = T_\theta(H^{\mathrm{in}}; M) \qquad \text{(式 6)}
$$

$M$ 掩蔽填充的组分位置，$\bar{h}_i$ 为组分的混合物条件表示，$h_{\mathrm{mix}}$ 为全局混合物表示。每层 Transformer 用标准多头自注意力：

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_h}} + M\right) V \qquad \text{(式 7)}
$$

RDKit/Uni-Mol/官能团信息通过融合分子 token 影响注意力，而非额外两两注意偏置。自注意力用所有共存物种信息更新每个组分。对任意组分置换 $\Pi$，编码器满足 $T_\theta([h_{\mathrm{mix}}^0, \Pi H^0])=[h_{\mathrm{mix}}, \Pi \bar{H}]$（式 8），因此组分表示置换等变、混合物表示置换不变，其中 $\bar{H}=[\bar{h}_1,\dots,\bar{h}_N]$。

### 4.4 状态条件的分子相互作用学习

多组分 Transformer 提供混合物感知组分表示 $\bar{h}_i$ 与全局混合物表示 $h_{\mathrm{mix}}$，在评估分子相互作用前进一步热力学状态条件化。组分 $i$ 的条件向量结合全局混合物上下文与温度、压力、液相组成：

$$
c_i = \left[\, h_{\mathrm{mix}} \,\|\, \tilde{T} \,\|\, \tilde{P} \,\|\, x_i \,\right] \qquad \text{(式 9)}
$$

其中 $\tilde{T}$、$\tilde{P}$ 为归一化温度与压力。条件编码器把 $c_i$ 映射到组分特异尺度与位移向量：

$$
(s_i, t_i) = f^{\mathrm{cond}}(c_i) \qquad \text{(式 10)}
$$

从而调制混合物感知表示：

$$
\tilde{h}_i = \bar{h}_i \odot \left[ 1 + \rho \tanh(s_i) \right] + t_i \qquad \text{(式 11)}
$$

对每个组分对 $(i,j)$，状态条件表示通过对称对算子组合：

$$
z_{ij} = \left[\, \tilde{h}_i + \tilde{h}_j \,\|\, |\tilde{h}_i - \tilde{h}_j| \,\|\, h_{\mathrm{mix}} \,\right] \qquad \text{(式 12)}
$$

共享对势网络把 $z_{ij}$ 映射到标量有效相互作用：

$$
I_{ij} = \phi_\theta(z_{ij}) \qquad \text{(式 13)}
$$

对称构造给出 $I_{ij}=I_{ji}$，允许同一相互作用网络跨所有组分对共享。因 $\bar{h}_i$、$\bar{h}_j$、$h_{\mathrm{mix}}$ 由所有组分联合编码，$I_{ij}$ 受完整混合物环境条件化。

### 4.5 过量吉布斯能与活度系数解码

ThermoFormer 通过无量纲摩尔过量吉布斯能表示液相非理想性：

$$
g^E \equiv \frac{G^E}{RT} = \sum_{1 \le i < j \le N} x_i x_j I_{ij} \qquad \text{(式 14)}
$$

二元：$g^E_{\mathrm{binary}} = x_1 x_2 I_{12}$（式 15）；三元：$g^E_{\mathrm{ternary}} = x_1 x_2 I_{12} + x_1 x_3 I_{13} + x_2 x_3 I_{23}$（式 16）。组成 prefactor 使 $g^E \to 0$ 在每个纯组分顶点恢复。与固定二元参数构造不同，式 (16) 的对项受完整三元环境条件化。活度系数由共享标量势经自动微分得到：

$$
\ln\gamma_i = g^E + \frac{\partial g^E}{\partial x_i} - \sum_j x_j \frac{\partial g^E}{\partial x_j} \qquad \text{(式 17)}
$$

所有组分活度系数通过同一过量吉布斯能表示耦合，而非由独立输出头预测。该构造通过共享可微势强制 Gibbs–Duhem 一致性。

### 4.6 纯组分蒸气压分支

组分 $i$ 的饱和蒸气压由其融合分子 token 与温度预测：

$$
\ln P_i^{\mathrm{sat}}(T) = f^{\mathrm{sat}}(h_i^0, T) \qquad \text{(式 18)}
$$

分支只依赖纯组分表示与温度，混合物效应通过 $\gamma_i$ 进入，从而把纯组分挥发性与液相相互作用效应分离。

### 4.7 可微气液相平衡重建

活度系数与饱和蒸气压通过修正的 Raoult 关系耦合：

$$
y_i P = x_i \gamma_i(T,P,\mathbf{x}) \, P_i^{\mathrm{sat}}(T) \qquad \text{(式 19)}
$$

低压理想汽相近似下，计算泡点压力：

$$
P^{\mathrm{calc}}(T,P,\mathbf{x}) = \sum_i x_i \gamma_i(T,P,\mathbf{x}) \, P_i^{\mathrm{sat}}(T) \qquad \text{(式 20)}
$$

相应汽相组成：

$$
y_i = \frac{x_i \gamma_i P_i^{\mathrm{sat}}}{\sum_j x_j \gamma_j P_j^{\mathrm{sat}}} \qquad \text{(式 21)}
$$

满足 $\sum_i y_i = 1$。等温任务给 $T$、$\mathbf{x}$，泡点压力满足隐式式 $P = P^{\mathrm{calc}}(T,P,\mathbf{x})$（式 22），因活度系数依赖压力，用阻尼定点迭代求解：

$$
P^{(k+1)} = (1-\eta)\,P^{(k)} + \eta\, P^{\mathrm{calc}}\!\left(T, P^{(k)}, \mathbf{x}\right) \qquad \text{(式 23)}
$$

其中 $\eta$ 为阻尼系数。等压任务给 $P^\ast$、$\mathbf{x}$，泡点温度由 $r(T) = P^{\mathrm{calc}}(T,P^\ast,\mathbf{x}) - P^\ast = 0$（式 24）确定，残差导数用中心差分估计：

$$
\frac{\partial r}{\partial T} \approx \frac{r(T+\Delta T) - r(T-\Delta T)}{2\,\Delta T} \qquad \text{(式 25)}
$$

用括号牛顿步更新温度：

$$
T_{N}^{(k+1)} = T^{(k)} - \frac{r(T^{(k)})}{\left.\partial r/\partial T\right|_{T^{(k)}}} \qquad \text{(式 26)}
$$

越界提议用括号中点替代。所有求解操作以可微张量计算实现，误差可在 VLE 求解器、蒸气压分支、过量吉布斯解码器、对势网络、Transformer 与分子表示模块间传播。

### 4.8 两阶段监督与逸度约束优化

第一阶段最小化泡点压力、泡点温度与汽相组成预测误差：

$$
\mathcal{L}_{\mathrm{sup}} = \lambda_P L_P + \lambda_T L_T + \lambda_y L_y \qquad \text{(式 27)}
$$

其中 $L_P$、$L_T$ 分别监督等温、等压泡点任务，$L_y$ 监督两种模式下的汽相组成。最佳验证 checkpoint 初始化物理微调阶段。在 VLE 处每组分液相与汽相逸度相等：

$$
f_i^{L} = f_i^{V} \qquad \text{(式 28)}
$$

低压理想汽相近似下分逸度：

$$
f_i^{L} = x_i \gamma_i P_i^{\mathrm{sat}}(T), \qquad f_i^{V} = y_i P \qquad \text{(式 29, 30)}
$$

平衡条件：

$$
x_i \gamma_i P_i^{\mathrm{sat}}(T) = y_i P \qquad \text{(式 31)}
$$

对实验样本 $n$，在观测状态评估组分级对数码-逸度残差：

$$
r_{n,i}^{f} = \ln x_{n,i}^{\mathrm{obs}} + \ln\hat{\gamma}_{n,i} + \ln \hat{P}_{n,i}^{\mathrm{sat}}(T_n^{\mathrm{obs}}) - \ln y_{n,i}^{\mathrm{obs}} - \ln P_n^{\mathrm{obs}}
$$

$$
= \ln\!\left[ \frac{x_{n,i}^{\mathrm{obs}}\,\hat{\gamma}_{n,i}\,\hat{P}_{n,i}^{\mathrm{sat}}(T_n^{\mathrm{obs}})}{y_{n,i}^{\mathrm{obs}}\,P_n^{\mathrm{obs}}} \right] \qquad \text{(式 32)}
$$

残差用实验 $T,P,x,y$ 而非 VLE 求解器生成（后者由构造满足式 31）。接近零组成的组分从对数残差中排除：

$$
m_{n,i} = \mathbb{I}\big(x_{n,i}^{\mathrm{obs}} > x_{\min}\big)\, \mathbb{I}\big(y_{n,i}^{\mathrm{obs}} > y_{\min}\big) \qquad \text{(式 33)}
$$

其中 $\mathbb{I}(\cdot)$ 为指示函数。用质量加权 Huber 惩罚计算逸度平衡损失：

$$
\mathcal{L}_{\mathrm{fug}} = \frac{\sum_n \sum_i w_n m_{n,i}\, \rho_\delta\!\left(r_{n,i}^{f}\right)}{\sum_n \sum_i w_n m_{n,i}} \qquad \text{(式 34)}
$$

其中 $w_n$ 为实验质量权重，$\rho_\delta$ 为 Huber 函数。物理微调目标：

$$
\mathcal{L}_{\mathrm{fine}} = \mathcal{L}_{\mathrm{sup}} + \lambda_{\mathrm{fug}}\, \mathcal{L}_{\mathrm{fug}} \qquad \text{(式 35)}
$$

保留 $\mathcal{L}_{\mathrm{sup}}$ 防止逸度正则化损伤直接评估的预测。因 $\mathcal{L}_{\mathrm{fug}}$ 约束乘积 $\gamma_i P_i^{\mathrm{sat}}$ 而非两因子独立，微调期间冻结蒸气压分支；分子投影器、多视图融合与 Transformer 编码器也固定，只微调对势网络、状态条件模块与混合物 token。

---

## 参考文献（要点，[编号] 对应原文）

1. Frenkel 等，ThermoML. J. Chem. Eng. Data 48, 2–13 (2003).
2. Renon & Prausnitz, NRTL. AIChE J. 14, 135–144 (1965).
3. Wilson, VLE 新过量自由能表达. JACS 86, 127–130 (1964).
4. Abrams & Prausnitz, UNIQUAC. AIChE J. 21, 116–128 (1975).
5. Fredenslund 等, UNIFAC. AIChE J. 21, 1086–1099 (1975).
6. Weidlich & Gmehling, modified UNIFAC. I&EC Res. 26, 1372–1381 (1987).
7. Constantinescu & Gmehling, UNIFAC Dortmund 6 版. J. Chem. Eng. Data 61, 2738–2748 (2016).
8. Klamt, COSMO-RS. JPCB 99, 2224–2235 (1995).
9. Lin & Sandler, COSMO-SAC. I&EC Res. 41, 899–913 (2002).
10–24. 机器学习/热力学 VLE 模型：ANN、描述符 ANN/RF、SMILES-RNN、GNN、ChemBERTa-2、GROVER、Uni-Mol、SolvGNN、SPT-NRTL、GDI-GNN、GE-GNN、HANNA、TeNNet-SAC 等（详见原文）。
25. Boiko 等, Autonomous chemical research with LLMs. Nature 624, 570–578 (2023).
26. Bran 等, Augmenting LLMs with chemistry tools. Nat. Mach. Intell. 6, 525–535 (2024).
27. Schäfer 等, agentic process design in flowsheet simulation. AIChE J. 70488 (2026).
28. Tan 等, reasoning-agent distillation simulation. Commun. Eng. 5, 26 (2026).
29. RDKit. https://www.rdkit.org (2026).
30. Ji 等, Uni-Mol v2（分子预训练模型）. NeurIPS 37 (2024).

---

*译文说明：论文 2.6、3 两节在 PDF 原文中未展开正文（仅标题），译文保留章节框架并标注；配套实验见 `ThermoFormer萃取精馏过程.md`。公式编号沿用原文式号。*
