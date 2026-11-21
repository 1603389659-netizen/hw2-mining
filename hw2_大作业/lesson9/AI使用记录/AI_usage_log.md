# HW2 AI 使用记录

> 课题：零售银行客户分群与客户价值识别（模拟数据）
> 原则：只记录**真实发送过的提示词**与**真实运行结果**；AI 生成的结论与本人验证分开标注。
> 第 9 课课堂作业的使用记录已归档至 `AI使用记录/AI_usage_log_lesson09.md`，未删除。

## 0. 项目与环境

- 工作目录：`D:\tare 文件\人工智能程序设计与实践\lesson9`
- 数据：`data/mall_customers_sim.csv`（**课程模拟数据**，300 行；从课堂演示包复制，MD5 一致，原文件未改）
- Python：`C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe`
- 依赖：pandas、numpy、matplotlib、scikit-learn
- 字体：Microsoft YaHei；随机过程固定 `random_state=42`

---

## 1. 起点指令：研究方案

### 1.1 真实提示词（原文）

```text
请为HW2设计完整的数据挖掘研究方案。

研究主题：
零售银行客户分群与客户价值识别。

目标读者：
银行零售业务部经理。

数据：
mall_customers_sim.csv。

请设计完整研究流程：

1. 业务问题与研究目标。
2. 数据初检与预处理。
3. 特征选择及理由。
4. 模型一：KMeans客户聚类。
5. 模型二：DBSCAN密度聚类。
6. 模型评估与结果比较。
7. 客户画像分析。
8. 业务结论与建议。
9. 局限性与改进方向。

要求：
- customer_id不得进入模型。
- 距离类模型必须StandardScaler标准化。
- KMeans必须显式设置random_state=42、n_init=10。
- 所有实验必须真实运行。
- 不得编造实验结果。

先生成研究方案：
reports/research_plan.md

等待我确认后再开始建模。
```

### 1.2 产出

- `reports/research_plan.md`：九部分研究流程 + 执行顺序、交付物、复现约定。
- 方案中不预设聚类结果，所有指标以实际运行为准。
- 后续步骤按此方案执行。

---

## 2. 数据预处理指令

### 2.1 真实提示词（原文）

```text
现在执行HW2数据预处理。

使用data/mall_customers_sim.csv。

要求：

1. 检查缺失值、重复值和异常值。
2. 选择具有业务意义的数值特征。
3. 排除customer_id、gender和city等非数值标识字段；如果需要使用分类字段，必须单独说明编码方法。
4. 使用StandardScaler标准化模型特征。
5. 保留原始数据，不覆盖原文件。
6. 输出预处理前后的统计信息。

生成：
generated/01_preprocessing.py
generated/cleaned_data.csv
generated/preprocessing_report.md

运行并检查结果。
```

### 2.2 实际运行结果

- 原始数据：300 行 × 7 列；缺失 0；完全重复 0；customer_id 重复 0（唯一值 300）。
- IQR 异常：age 0、income 22、balance 43、product_count 2；
  至少命中一项 50 行（16.67%）。
- 业务范围异常：balance 负值 10 条，最小 -0.73 万元。
- **处理：标记但不删除**（高值可能是真实高价值客户；负余额在少量范围内可解释为透支）。
- 特征：age、income、balance、product_count；排除 customer_id、gender、city
  （分类字段需 one-hot，但会造成维度膨胀与距离失真，本次不使用）。
- 标准化后四列均值≈0、样本标准差≈1.0017。
- 输出：`generated/cleaned_data.csv`（300×11）、`generated/preprocessing_report.md`。

---

## 3. KMeans 建模指令

### 3.1 真实提示词（原文）

```text
完成HW2第一个模型：KMeans客户聚类。

要求：

1. 使用预处理后的数据。
2. 使用StandardScaler处理后的特征。
3. 使用KMeans进行聚类。
4. 设置random_state=42、n_init=10。
5. 比较不同K值。
6. 使用肘部法和轮廓系数辅助选择K。
7. 输出每个簇的人数和占比。
8. 生成客户画像。
9. 绘制中文聚类结果图。

生成：
generated/02_kmeans.py
generated/kmeans_results.csv
generated/kmeans_report.md
outputs/01_kmeans.png

实际运行并打印结果。
```

### 3.2 实际运行结果

| K | inertia_ | 轮廓系数 | 各簇人数 |
|---|---|---|---|
| 2 | 642.8650 | 0.514975 | 247 / 53 |
| 3 | 436.8879 | 0.402186 | 111 / 138 / 51 |
| 4 | 323.2085 | 0.412272 | 50 / 54 / 102 / 94 |
| 5 | 262.5755 | 0.419650 | 72 / 49 / 91 / 44 / 44 |
| 6 | 236.2191 | 0.397908 | 57 / 49 / 41 / 58 / 71 / 24 |
| 7 | 206.6604 | 0.372910 | 24 / 71 / 57 / 28 / 58 / 22 / 40 |
| 8 | 188.0984 | 0.361380 | 57 / 24 / 53 / 22 / 17 / 26 / 58 / 43 |
| 9 | 169.7410 | 0.367286 | 17 / 56 / 14 / 41 / 22 / 17 / 58 / 54 / 21 |

- 几何肘部：**K=4**；K≥3 轮廓系数最优 K=5（0.419650）。
- 选 K 规则：K=2 虽轮廓系数最高（0.514975），但二分会硬合并不同客群；
  在 K≥3 中取"轮廓系数 ≥ 最优值 98%"的最小 K → **K=4**（0.412272），与肘部一致。
- 最终：K=4，各簇 50 / 54 / 102 / 94，合计 300，inertia_=323.2085。
- 输出：`kmeans_results.csv`（300 行）、`kmeans_report.md`、`outputs/01_kmeans.png`。

---

## 4. DBSCAN 建模指令

### 4.1 真实提示词（原文）

```text
完成HW2第二个模型：DBSCAN客户聚类。

要求：

1. 使用与KMeans相同的预处理数据。
2. 使用StandardScaler标准化后的特征。
3. 使用DBSCAN进行聚类。
4. 合理尝试不同eps和min_samples参数。
5. 输出聚类数量、各簇人数和噪声点数量。
6. 绘制中文聚类结果图。
7. 分析DBSCAN与KMeans的区别。
8. 不得编造实验结果。

生成：
generated/03_dbscan.py
generated/dbscan_results.csv
generated/dbscan_report.md
outputs/02_dbscan.png

实际运行程序并打印结果。
```

### 4.2 实际运行结果

- 网格：eps ∈ [0.8, 2.2]（步长 0.1）× min_samples ∈ [3, 10] = **120 组**。
- 满足"2~6 簇、噪声 ≤15%"：**36 组**。
- 最优（非噪声轮廓系数最高）：**eps=1.0、min_samples=7**，轮廓系数 0.541996。
- 结果：**2 个簇（242 / 43）+ 15 个噪声（5.00%）**，覆盖率 95.00%。
- 噪声客户画像：平均年龄 43.93 岁、收入 30.85 万元、余额 31.87 万元、产品数 3.13，
  明显高于整体，属于"高价值但特征偏离主体"的客户。
- eps>1.2 后退化为 1 簇，参数敏感性明显。
- 输出：`dbscan_results.csv`、`dbscan_grid.csv`（120 组）、`dbscan_report.md`、`outputs/02_dbscan.png`。

---

## 5. 模型评估指令

### 5.1 真实提示词（原文）

```text
现在完成HW2模型评估。

比较KMeans和DBSCAN。

要求：

1. 计算适用的轮廓系数。
2. 统计各模型聚类数量。
3. 统计噪声点数量。
4. 比较各簇规模。
5. 分析两种算法的优点和局限。
6. 解释哪个模型更适合当前客户数据。
7. 不要只根据单一指标决定模型优劣。

生成：
generated/04_model_evaluation.py
generated/model_comparison.csv
generated/model_comparison.md
outputs/03_model_comparison.png

输出真实计算结果。
```

### 5.2 实际运行结果

| 指标 | KMeans | DBSCAN |
|---|---|---|
| 聚类数量 | 4 | 2 |
| 噪声点 | 0 | 15（5.00%） |
| 覆盖率 | 100.00% | 95.00% |
| 轮廓系数 | 0.412272（300 样本） | 0.541996（285 非噪声） |
| 最大/最小簇 | 102 / 50（2.04） | 242 / 43（5.63） |

- **ARI = 0.231501**。
- 交叉表：KMeans 簇 0（50 人）几乎整体对应 DBSCAN 簇 1（43 人）+7 个噪声；
  KMeans 簇 1/2/3 基本落入 DBSCAN 簇 0，其中 2/6 个点被判为噪声。
- 结论：主分群用 KMeans（100% 覆盖、4 簇均衡、可解释），DBSCAN 作异常/高价值边缘客户识别；
  不因 DBSCAN 轮廓系数更高就判定其更优（口径不同：2 簇 vs 4 簇、剔除噪声、覆盖 95%）。

---

## 6. 业务分析指令

### 6.1 真实提示词（原文）

```text
请基于HW2真实实验结果，形成零售银行客户分群业务分析。

要求：

1. 总结不同客户群体的特征。
2. 分析各群体的收入、余额、年龄及产品持有情况。
3. 为不同客群起中文业务名称。
4. 提出差异化营销和客户维护建议。
5. 说明模型可能存在的误分类和局限。
6. 不得把模拟数据结论描述为真实银行经营结论。

生成：
generated/05_business_analysis.py
generated/customer_segments.csv
generated/business_analysis.md
outputs/04_customer_segments.png
```

### 6.2 实际运行结果

| 簇 | 客群名称 | 人数 | 占比 | 年龄 | 收入(万元) | 余额(万元) | 产品数 |
|---|---|---|---|---|---|---|---|
| 0 | 高净值中年客群 | 50 | 16.67% | 47.10 | 41.58 | 48.35 | 1.96 |
| 1 | 年轻多产品成长客群 | 54 | 18.00% | 23.93 | 9.32 | 5.12 | 3.93 |
| 2 | 中年稳健理财客群 | 102 | 34.00% | 39.55 | 19.29 | 12.23 | 1.95 |
| 3 | 年轻低净值潜力客群 | 94 | 31.33% | 21.33 | 6.88 | 2.15 | 1.13 |

命名由"各簇均值 vs 全局均值"的规则生成（脚本 `name_segment`），不是按簇编号写死。
建议与局限见 `generated/business_analysis.md` 第四节，明确标注为模拟数据结论。

---

## 7. 材料汇总与 AI 记录指令

### 7.1 真实提示词（原文，两段连续发出）

```text
请根据本项目实际运行结果，整理HW2报告所需的全部材料。

按照老师aip-major-assignment-2.docx的六部分结构：

1. 问题与目标。
2. 方法与过程。
3. 实验结果。
4. 结论与建议。
5. 局限与改进。
6. AI使用记录。

请提取：
- 实际样本量。
- 特征选择。
- 预处理方法。
- 两种模型的参数。
- 模型评估指标。
- 客户分群统计结果。
- 图表文件名。
- 业务分析结果。
- AI使用记录。

生成：
reports/hw2_materials.md

不要自行编造结果。
暂时不要写最终报告，等待检查。
```

```text
请整理HW2完整AI使用记录。

生成：
AI使用记录/AI_usage_log.md

记录：
1. 起点指令。
2. 数据预处理指令。
3. KMeans建模指令。
4. DBSCAN建模指令。
5. 模型评估指令。
6. 业务分析指令。
7. 修改和纠错指令。
8. 实际运行验证过程。

必须使用真实执行过的指令和结果，不得虚构。
```

### 7.2 产出

- `reports/hw2_materials.md`：样本量、特征、预处理、参数、指标、分群统计、图表文件名、
  业务分析、AI 记录索引，以及"待确认事项"。
- 本文即 HW2 的 AI 使用记录。
- 尚未撰写最终报告，等待本人确认。

---

## 8. 修改与纠错记录（真实发生）

> 本次未收到用户单独发送的"纠错提示词"，以下为执行过程中实际出现的问题与修正，
> 均为真实运行日志，不是补写。

### 8.1 预处理脚本首次运行

- 无报错。但输出显示 IQR 异常达 50 行（16.67%），若直接删除会损失六分之一客户，
  故在报告中明确"标记不删除"，并把该比例写入局限。

### 8.2 KMeans 脚本首次运行失败

- 报错：`ValueError: Both input arrays must be (arrays of) 3-dimensional vectors, but they are 2 and 2 dimensional instead.`
- 原因：numpy 2.5 中 `np.cross` 不再支持二维向量，肘部距离计算报错。
- 修正：改为手工二维叉积 `|dx1*dy2 − dy1*dx2| / |p2−p1|`，并先对 K 与 SSE 归一化，
  避免坐标量纲影响"点到直线距离"。
- 重跑后正常，几何肘部 K=4，与轮廓系数规则一致。

### 8.3 业务分析脚本运行前自查

- 发现 Z 值计算用 `f"平均{特征名}"` 拼列名会生成"平均age"这类不存在的列。
- 修正：显式列映射 `{"age": "平均年龄", ...}` 后再取数。
- 运行通过，热力图四簇 × 四特征均有真实 Z 值。

### 8.4 研究（非 HW2）中曾出现的同类问题

- 第 9 课肘部脚本与画图脚本的字体、列错位问题已在旧记录中说明，HW2 预处理直接使用 Windows 字体，
  未再复现。

---

## 9. 实际运行与验证过程

### 9.1 执行命令（均在项目根目录）

```text
& "C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe" generated/01_preprocessing.py
& "C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe" generated/02_kmeans.py
& "C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe" generated/03_dbscan.py
& "C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe" generated/04_model_evaluation.py
& "C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe" generated/05_business_analysis.py
```

### 9.2 核对点（已实际检查）

1. `data/` 原始 CSV 与课堂演示包文件 MD5 一致（B1A0F6286BCE8BFF3D297B0979A74C58），未修改。
2. 聚类人数合计 = 300；KMeans 四簇 50+54+102+94 = 300。
3. 特征矩阵只含 `z_age、z_income、z_balance、z_product_count`，无 customer_id。
4. `cleaned_data.csv` 为 300 行 × 11 列；`kmeans_results.csv` 300 行；`dbscan_results.csv` 300 行。
5. 四张 PNG 已逐张打开检查：中文正常、图例人数与终端一致、无空图、噪声点标记可见。
6. DBSCAN 结果未美化：2 簇 + 15 噪声按实际输出写入报告，未改成"4 簇"以迎合 KMeans。

### 9.3 图表文件

| 文件 | 已查看 | 备注 |
|---|---|---|
| `outputs/01_kmeans.png` | 是 | 肘部 K=4 标线、轮廓 K=4 标线、四簇散点 |
| `outputs/02_dbscan.png` | 是 | 2 簇 + 15 噪声，eps 敏感性曲线在 1.0 处峰值 |
| `outputs/03_model_comparison.png` | 是 | 两模型散点 + 指标柱状（口径标注） |
| `outputs/04_customer_segments.png` | 是 | 规模、Z 值热力图、客群分布 |

### 9.4 尚未由本人完成的验证

- [ ] 本人逐张复核四张 PNG 与报告中的数字
- [ ] 按老师 docx 模板核对章节、图表编号与格式
- [ ] 最终报告尚未撰写（按要求等待确认）

---

## 10. 说明与边界

1. 本文提示词均为实际发送的原文，未改写或补造。
2. 所有指标来自程序输出；`reports/hw2_materials.md` 中的数字与 CSV/终端一致。
3. 明确声明：数据为课程模拟数据，分群名称与建议仅为方法演示，不代表真实银行经营结论。
4. 第 9 课旧日志保留为 `AI_usage_log_lesson09.md`，未覆盖。
