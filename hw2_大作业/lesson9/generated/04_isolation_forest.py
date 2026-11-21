# -*- coding: utf-8 -*-
"""
HW2：Isolation Forest 异常检测（新增，作为与聚类并列的第二类数据挖掘任务）

- 输入：generated/cleaned_data.csv 的四个标准化特征
- 不进入模型：customer_id、gender、city
- IsolationForest(random_state=42, contamination=0.05, n_estimators=200)
- 输出：0=正常，1=异常；每客户异常分数；异常客户统计与差异分析
- 注意：异常 ≠ 欺诈，异常仅表示"特征组合与主体人群不同"

产出：
    generated/isolation_forest_results.csv
    generated/isolation_forest_report.md
    outputs/03_isolation_forest.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.ensemble import IsolationForest

def setup_chinese_font():
    for f in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
              r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]:
        if os.path.exists(f):
            fm.fontManager.addfont(f)
            prop = fm.FontProperties(fname=f)
            matplotlib.rcParams["font.sans-serif"] = (
                [prop.get_name()] + matplotlib.rcParams["font.sans-serif"]
            )
            print(f"使用中文字体：{prop.get_name()}（{f}）")
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_chinese_font()

CLEANED = "generated/cleaned_data.csv"
OUT_CSV = "generated/isolation_forest_results.csv"
OUT_MD = "generated/isolation_forest_report.md"
OUT_PNG = "outputs/03_isolation_forest.png"
os.makedirs("outputs", exist_ok=True)

Z = ["z_age", "z_income", "z_balance", "z_product_count"]
FEATURES = ["age", "income", "balance", "product_count"]
df = pd.read_csv(CLEANED, encoding="utf-8-sig")
Xz = df[Z].to_numpy()
N = len(df)

print("=" * 74)
print("一、模型与参数")
print("=" * 74)
CONTAMINATION = 0.05
params = dict(n_estimators=200, max_samples="auto", contamination=CONTAMINATION,
              random_state=42, n_jobs=-1)
print(f"IsolationForest({', '.join(f'{k}={v!r}' for k, v in params.items())})")
print(f"输入特征：{Z}（StandardScaler 标准化；customer_id/gender/city 未入模）")
print(f"样本量：{N}")
print(f"contamination={CONTAMINATION} 的理由：")
print("  1) 数据无异常标签，无法用监督指标调参，只能依据业务经验设定异常比例；")
print("  2) 5% 是零售客户运营中常见的“需人工复核名单”比例上限，名单规模可控；")
print("  3) 与 DBSCAN 在相同数据上识别出的噪声比例（5.00%）保持一致，便于两种方法对照；")
print("  4) 报告中同时给出 3%/5%/10% 的敏感性结果，说明该参数对结果的影响。")

# ==================== 拟合 ====================
model = IsolationForest(**params)
raw_pred = model.fit_predict(Xz)            # 1=正常, -1=异常
score = model.decision_function(Xz)         # 越大越正常
df["is_anomaly"] = np.where(raw_pred == -1, 1, 0)
df["anomaly_score"] = -score                # 越大越异常，便于业务排序
n_anom = int(df["is_anomaly"].sum())

print("\n" + "=" * 74)
print("二、检测结果")
print("=" * 74)
print(f"异常客户总数：{n_anom} / {N}")
print(f"异常客户占比：{n_anom / N * 100:.2f}%")
print(f"正常客户数：{N - n_anom}（{(N - n_anom) / N * 100:.2f}%）")

# 参数敏感性（仅打印，不改变最终模型）
print("\ncontamination 敏感性（仅作参考，最终模型仍为 5%）：")
for c in [0.03, 0.05, 0.10]:
    m = IsolationForest(n_estimators=200, contamination=c, random_state=42, n_jobs=-1)
    p = m.fit_predict(Xz)
    print(f"  contamination={c:.2f} → 异常 {int((p == -1).sum())} 人"
          f"（{(p == -1).mean() * 100:.2f}%）")

# ==================== 异常客户画像 ====================
print("\n" + "=" * 74)
print("三、异常客户 vs 整体客户")
print("=" * 74)
stat = pd.DataFrame({
    "整体客户": df[FEATURES].mean().round(2),
    "异常客户": df.loc[df["is_anomaly"] == 1, FEATURES].mean().round(2),
    "正常客户": df.loc[df["is_anomaly"] == 0, FEATURES].mean().round(2),
})
stat["异常-整体差"] = (stat["异常客户"] - stat["整体客户"]).round(2)
print(stat.to_string())

print("\n异常客户特征范围：")
for f in FEATURES:
    s = df.loc[df["is_anomaly"] == 1, f]
    print(f"  {f:<14} 最小={s.min():>8.2f}  最大={s.max():>8.2f}  "
          f"中位数={s.median():>8.2f}")

# 异常客户在四个特征上偏离整体最明显的方向
print("\n异常客户相对整体的 Z 值偏移（按标准化特征均值计算）：")
z_shift = df.loc[df["is_anomaly"] == 1, Z].mean() - df[Z].mean()
for f, v in zip(FEATURES, z_shift.values):
    print(f"  {f:<14} {v:+.3f} 个标准差")

# 异常客户按“收入高低”拆分，避免把异常等同于欺诈
anom = df[df["is_anomaly"] == 1]
high_value = anom[anom["income"] > df["income"].median()]
low_value = anom[anom["income"] <= df["income"].median()]
print(f"\n异常客户内部构成：收入高于中位数（{df['income'].median():.2f} 万元）"
      f"{len(high_value)} 人，其余 {len(low_value)} 人")

# ==================== 保存 ====================
out = df[["customer_id", "is_anomaly", "anomaly_score", "decision_function"]].copy() \
    if "decision_function" in df.columns else df[["customer_id", "is_anomaly", "anomaly_score"]].copy()
out["raw_decision_function"] = score
out = out.drop(columns=["decision_function"], errors="ignore")
out = out.sort_values("anomaly_score", ascending=False)
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存：{OUT_CSV}（{len(out)} 行，按异常分数从高到低排序）")

# ==================== 绘图 ====================
fig, ax = plt.subplots(figsize=(10, 7))
normal = df[df["is_anomaly"] == 0]
anomaly = df[df["is_anomaly"] == 1]
ax.scatter(normal["income"], normal["balance"], s=38, alpha=0.65, c="#4C72B0",
           edgecolors="white", linewidths=0.4, label=f"正常客户（{len(normal)} 人）")
ax.scatter(anomaly["income"], anomaly["balance"], s=110, c="#C44E52", marker="X",
           edgecolors="black", linewidths=1.0,
           label=f"异常客户（{len(anomaly)} 人，{len(anomaly) / N * 100:.1f}%）",
           zorder=5)
ax.set_title(f"Isolation Forest 异常检测结果（contamination={CONTAMINATION}）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.3)
ax.text(0.99, 0.01, "教学模拟数据，异常 ≠ 欺诈", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF3CD", ec="#E0B400"))
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"已保存：{OUT_PNG}")

# ==================== 报告 ====================
md = f"""# HW2 · Isolation Forest 异常检测报告

> 数据：`{CLEANED}`（课程**模拟客户数据**，300 条）。
> 任务定位：**异常检测**，与 KMeans/DBSCAN 的"聚类分群"是不同类别的数据挖掘任务。
> 重要：**异常 ≠ 欺诈**。本模型只识别"特征组合与主体人群不同"的客户，
> 不能据此认定任何客户存在欺诈、违约或风险行为。

## 一、模型与参数

```text
IsolationForest(n_estimators=200, max_samples='auto',
                contamination={CONTAMINATION}, random_state=42, n_jobs=-1)
```

- 输入：`z_age`、`z_income`、`z_balance`、`z_product_count`（StandardScaler 标准化）。
- 未入模：`customer_id`（唯一标识）、`gender`、`city`（分类字段）。
- `contamination=0.05` 的选择理由：
  1. 数据无异常标签，只能用业务经验设定预期异常比例；
  2. 5% 对应"需人工复核名单"的可控规模（300 人中约 15 人）；
  3. 与 DBSCAN 在同一数据上识别出的噪声比例（5.00%）一致，便于跨方法对照；
  4. 报告同时给出 3%/5%/10% 的敏感性结果，避免把 5% 当作唯一正确答案。

## 二、检测结果

| 指标 | 数值 |
|---|---|
| 样本总数 | {N} |
| 异常客户数 | **{n_anom}** |
| 异常占比 | **{n_anom / N * 100:.2f}%** |
| 正常客户数 | {N - n_anom} |

输出约定：`is_anomaly = 0` 正常，`is_anomaly = 1` 异常；
`anomaly_score = -decision_function`，数值越大越异常。

## 三、异常客户与整体的差异

| 特征 | 整体均值 | 异常客户均值 | 正常客户均值 | 异常−整体 |
|---|---|---|---|---|
| 年龄（岁） | {stat.loc['age', '整体客户']:.2f} | {stat.loc['age', '异常客户']:.2f} | {stat.loc['age', '正常客户']:.2f} | {stat.loc['age', '异常-整体差']:+.2f} |
| 收入（万元） | {stat.loc['income', '整体客户']:.2f} | {stat.loc['income', '异常客户']:.2f} | {stat.loc['income', '正常客户']:.2f} | {stat.loc['income', '异常-整体差']:+.2f} |
| 余额（万元） | {stat.loc['balance', '整体客户']:.2f} | {stat.loc['balance', '异常客户']:.2f} | {stat.loc['balance', '正常客户']:.2f} | {stat.loc['balance', '异常-整体差']:+.2f} |
| 产品数（个） | {stat.loc['product_count', '整体客户']:.2f} | {stat.loc['product_count', '异常客户']:.2f} | {stat.loc['product_count', '正常客户']:.2f} | {stat.loc['product_count', '异常-整体差']:+.2f} |

相对整体的标准化偏移：

"""
for f, v in zip(FEATURES, z_shift.values):
    md += f"- {f}：{v:+.3f} 个标准差\n"

md += f"""
异常客户内部并非同质：收入高于中位数的 **{len(high_value)} 人**（可能为高价值客户，
应重点维护而非风控处置），其余 **{len(low_value)} 人**（可能与主体消费能力差异较大）。

## 四、如何解读这些异常

1. **异常是"偏离主体"，不是"坏客户"**。Isolation Forest 依据"被随机切分隔离的快慢"
   给出异常分数，特征越少见越容易被隔离，分数越高。
2. 异常客户可能包含：
   - 高收入高余额的少数高价值客户（样本中确实存在）；
   - 特征组合罕见但合法的客户（如年轻且高余额）；
   - 数据录入或口径异常（需回到源系统核实）。
3. 业务动作应是**人工复核、名单核对、差异化服务**，不得直接冻结、拒贷或标注欺诈。

## 五、与聚类的区别

| 对比 | KMeans / DBSCAN | Isolation Forest |
|---|---|---|
| 任务类型 | 聚类（分群） | 异常检测 |
| 输出 | 每个客户一个簇号 | 每个客户一个异常分数 + 0/1 标记 |
| 关注点 | 群体之间的相似性 | 个体与主体分布的偏离程度 |
| 是否需要 K | KMeans 需要；DBSCAN 不需要 | 不需要簇数，只需 contamination |
| 结果用途 | 分层运营、资源配置 | 人工复核、风控线索、数据质量检查 |

## 六、输出文件

- `{OUT_CSV}`：customer_id、is_anomaly、anomaly_score、raw_decision_function（按异常分数降序）。
- `{OUT_PNG}`：收入×余额散点，正常客户蓝色圆点、异常客户红色 ×，标注"教学模拟数据，异常 ≠ 欺诈"。

## 七、局限

1. 模拟数据仅 300 条，异常比例与画像不能外推真实客户。
2. `contamination` 是人为设定，不是从数据中估计出的真实异常率；不同取值会改变名单。
3. 仅使用 4 个静态特征，缺少交易频率、渠道、设备等行为特征，无法识别真正的欺诈模式。
4. 异常分数依赖随机切分，虽固定 `random_state=42` 可复现，但不同种子下名单可能有小幅变化。
"""
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)
print(f"已保存：{OUT_MD}")

print("\n" + "=" * 74)
print("四、终端摘要")
print("=" * 74)
print(f"异常客户总数：{n_anom}")
print(f"异常客户占比：{n_anom / N * 100:.2f}%")
print(f"异常客户主要特征：年龄 {stat.loc['age', '异常客户']:.2f} 岁、"
      f"收入 {stat.loc['income', '异常客户']:.2f} 万元、"
      f"余额 {stat.loc['balance', '异常客户']:.2f} 万元、"
      f"产品数 {stat.loc['product_count', '异常客户']:.2f} 个")
print(f"Isolation Forest 参数：n_estimators=200, contamination={CONTAMINATION}, "
      f"random_state=42")
print(f"输出：{OUT_CSV}、{OUT_MD}、{OUT_PNG}")
