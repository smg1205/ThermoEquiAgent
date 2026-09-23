# 方法学

第一个后端采用**修正 Raoult 定律**，配合理想活度系数与 Antoine 蒸气压。泡点求解 `sum(x_i P_i^sat(T)) = P`；露点求解 `sum(y_i P / P_i^sat(T)) = 1`。等温曲线求取泡点压力与汽相组成。TP 闪蒸使用 `K_i = P_i^sat/P` 求解 Rachford-Rice，并显式处理单相边界。

Wilson、NRTL、UNIQUAC 与 Peng-Robinson 通过模型卡和路由规则表示。它们**不会被理想适配器悄悄近似替代**。带参数的调用必须先解析出有证据支撑的参数，实现才允许执行。
