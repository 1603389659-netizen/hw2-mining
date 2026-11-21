# -*- coding: utf-8 -*-
"""
HW2 第 1 步：数据预处理

- 检查缺失值、重复值、异常值（IQR 1.5 倍法则 + 业务范围）
- 选择四个业务数值特征：age、income、balance、product_count
- 排除 customer_id / gender / city
- StandardScaler 标准化
- 保留原始数据，另存 cleaned_data.csv（原始列 + z_ 前缀标准化列）
- 输出预处理报告
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW = "data/mall_customers_sim.csv"
OUT_CSV = "generated/cleaned_data.csv"
OUT_MD = "generated/preprocessing_report.md"
os.makedirs("generated", exist_ok=True)

FEATURES = ["age", "income", "balance", "product_count"]

df = pd.read_csv(RAW, encoding="utf-8-sig")
n0, p0 = df.shape
print("=" * 70)
print("一、原始数据概览")
print("=" * 70)
print(f"文件：{RAW}")
print(f"形状：{n0} 行 × {p0} 列")
print(f"字段：{list(df.columns)}")
print("\n数据类型：")
print(df.dtypes.to_string())

# ==================== 缺失值 ====================
print("\n" + "=" * 70)
print("二、缺失值检查")
print("=" * 70)
miss = df.isna().sum()
print(miss.to_string())
print(f"缺失值总数：{int(miss.sum())}")

# ==================== 重复值 ====================
print("\n" + "=" * 70)
print("三、重复值检查")
print("=" * 70)
dup_rows = int(df.duplicated().sum())
dup_id = int(df["customer_id"].duplicated().sum())
print(f"完全重复行：{dup_rows}")
print(f"customer_id 重复：{dup_id}")
print(f"customer_id 唯一值：{df['customer_id'].nunique()} / {n0}")

# ==================== 异常值 ====================
print("\n" + "=" * 70)
print("四、异常值检查（IQR 1.5 倍法则 + 业务范围）")
print("=" * 70)
anomaly_summary = []
mask_anomaly = pd.Series(False, index=df.index)
for col in FEATURES:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = (df[col] < lo) | (df[col] > hi)
    mask_anomaly |= out
    print(f"{col:<14} Q1={q1:>9.2f} Q3={q3:>9.2f} 下限={lo:>9.2f} 上限={hi:>9.2f} "
          f"IQR异常={int(out.sum()):>3d}")
    anomaly_summary.append({
        "字段": col, "Q1": round(q1, 2), "Q3": round(q3, 2),
        "下限": round(lo, 2), "上限": round(hi, 2),
        "IQR异常数": int(out.sum()),
        "业务范围异常数": 0,
    })

# 业务规则：年龄 18~100；income>0；balance>=0；product_count 0~20
biz = {
    "age": ((df["age"] < 18) | (df["age"] > 100)),
    "income": (df["income"] <= 0),
    "balance": (df["balance"] < 0),
    "product_count": ((df["product_count"] < 0) | (df["product_count"] > 20)),
}
for i, col in enumerate(FEATURES):
    n_biz = int(biz[col].sum())
    anomaly_summary[i]["业务范围异常数"] = n_biz
    if n_biz:
        print(f"{col:<14} 业务范围异常：{n_biz} 条 "
              f"（取值示例：{sorted(df.loc[biz[col], col].unique())[:5]}）")
print(f"至少命中一项 IQR 异常的行数：{int(mask_anomaly.sum())}")
print(f"异常行占比：{mask_anomaly.mean() * 100:.2f}%")
print("处理策略：仅标记，不删除。高收入/高余额可能是真实高价值客户，")
print("          负余额（-7300 元以内）为教学模拟数据的少量透支值，保留并在报告中说明。")

# ==================== 字段选择说明 ====================
print("\n" + "=" * 70)
print("五、特征选择")
print("=" * 70)
print(f"进入模型：{FEATURES}")
print("排除：customer_id（唯一标识，无业务含义，进入模型会引入噪声）")
print("排除：gender、city（分类字段，欧氏距离不能直接使用；")
print("      如需使用应 one-hot 编码，但会造成维度膨胀与距离失真，本次不使用）")

# ==================== 标准化 ====================
print("\n" + "=" * 70)
print("六、StandardScaler 标准化")
print("=" * 70)
scaler = StandardScaler()
X = df[FEATURES].to_numpy()
Xz = scaler.fit_transform(X)
z_cols = [f"z_{c}" for c in FEATURES]
for i, c in enumerate(FEATURES):
    print(f"z_{c:<14} 均值={Xz[:, i].mean(): .6f}  标准差(样本)={Xz[:, i].std(ddof=1):.6f}  "
          f"原始均值={df[c].mean():.4f}  原始标准差={df[c].std(ddof=1):.4f}")

# ==================== 保存 ====================
out = df.copy()
for i, c in enumerate(z_cols):
    out[c] = Xz[:, i]
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存：{OUT_CSV}（{out.shape[0]} 行 × {out.shape[1]} 列，原始列保留 + 4 个 z_ 列）")

before_after = pd.DataFrame({
    "字段": FEATURES,
    "预处理前均值": [df[c].mean().round(4) for c in FEATURES],
    "预处理前标准差": [df[c].std(ddof=1).round(4) for c in FEATURES],
    "预处理后均值": [Xz[:, i].mean().round(6) for i in range(4)],
    "预处理后标准差": [Xz[:, i].std(ddof=1).round(6) for i in range(4)],
})

md = f"""# HW2 数据预处理报告

> 数据：`{RAW}`（课程提供的**模拟客户数据**，非真实银行业务数据）。
> 原始文件未修改，预处理结果另存为 `{OUT_CSV}`。

## 一、数据规模与字段

- 样本量：**{n0} 行 × {p0} 列**
- 字段：`{'`、`'.join(df.columns)}`
- 数值字段：age、income、balance、product_count
- 标识/分类字段：customer_id、gender、city

## 二、缺失值与重复值

| 检查项 | 结果 |
|---|---|
| 各列缺失值 | {'；'.join(f'{c}={int(v)}' for c, v in miss.items())} |
| 缺失值总数 | {int(miss.sum())} |
| 完全重复行 | {dup_rows} |
| customer_id 重复 | {dup_id} |
| customer_id 唯一值数 | {df['customer_id'].nunique()} |

结论：数据完整，无需删除或填补。

## 三、异常值检查

采用 IQR 1.5 倍法则，并叠加业务合理范围（年龄 18~100、收入>0、余额≥0、产品数 0~20）。

| 字段 | Q1 | Q3 | 下限 | 上限 | IQR 异常数 | 业务范围异常数 |
|---|---|---|---|---|---|---|
"""
for a in anomaly_summary:
    md += (f"| {a['字段']} | {a['Q1']} | {a['Q3']} | {a['下限']} | {a['上限']} | "
           f"{a['IQR异常数']} | {a['业务范围异常数']} |\n")

md += f"""
- 至少命中一项 IQR 异常的行数：**{int(mask_anomaly.sum())}**（占 {mask_anomaly.mean() * 100:.2f}%）。
- 业务范围异常：`balance` 存在少量负值，最小值 {df['balance'].min():.2f} 万元，
  为模拟数据中的透支/欠款情形。
- **处理策略：标记但不删除**。高收入高余额可能是真实高价值客户，删除会损失重要客群；
  负余额在少量范围内可解释为透支。上述判断在报告中明确说明，不当作脏数据静默处理。

## 四、特征选择

**进入模型（4 个数值特征）**：`age`、`income`、`balance`、`product_count`

| 特征 | 业务含义 | 选择理由 |
|---|---|---|
| age | 人生阶段 | 影响风险偏好与产品需求 |
| income | 收入能力 | 决定可支配资金与授信空间 |
| balance | 账户资产 | 反映本行资金沉淀 |
| product_count | 产品持有广度 | 反映关系深度与交叉销售空间 |

**排除字段**：

- `customer_id`：唯一标识，无业务含义；进入模型会把"编号大小"当成相似性，属于噪声甚至泄漏。
- `gender`、`city`：分类字段，KMeans/DBSCAN 依赖欧氏距离，不能直接计算。
  如确需使用，应单独说明编码方式（如 one-hot 或目标编码），并评估维度膨胀与距离失真；
  本次不纳入模型，仅可在分群后作描述性交叉分析。

## 五、标准化

对四个特征使用 `StandardScaler`（按列减均值、除以标准差），标准化只改变距离计算的度量衡，
不改变任何客户的原始业务数值。

| 字段 | 预处理前均值 | 预处理前标准差 | 预处理后均值 | 预处理后标准差 |
|---|---|---|---|---|
"""
for _, r in before_after.iterrows():
    md += (f"| {r['字段']} | {r['预处理前均值']} | {r['预处理前标准差']} | "
           f"{r['预处理后均值']} | {r['预处理后标准差']} |\n")

md += f"""
标准化后各列均值≈0、样本标准差≈1，四个特征在距离计算中等权。

## 六、输出文件

- `{OUT_CSV}`：{out.shape[0]} 行 × {out.shape[1]} 列，保留原始字段并追加 `z_age`、`z_income`、
  `z_balance`、`z_product_count` 四列标准化特征。
- 后续模型只使用 `z_` 前缀列；`customer_id` 仅用于结果表关联，不进入特征矩阵。

## 七、说明

- 本次数据不存在缺失与重复，预处理的主要工作是异常值识别、字段筛选与标准化。
- 未对异常值做截断或删除，避免在无业务依据的情况下改动客户数据；该局限在最终报告中说明。
"""
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)
print(f"已保存：{OUT_MD}")
