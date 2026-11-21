# -*- coding: utf-8 -*-
"""
HW2 第 4 步：模型评估与比较（KMeans vs DBSCAN）

- 轮廓系数（KMeans 全样本；DBSCAN 非噪声样本，需注明口径不同）
- 聚类数量、噪声数量、簇规模
- 两模型划分一致性（ARI）
- 不以单一指标定优劣
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import silhouette_score, adjusted_rand_score

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

df = pd.read_csv("generated/cleaned_data.csv", encoding="utf-8-sig")
km_df = pd.read_csv("generated/kmeans_results.csv", encoding="utf-8-sig")
db_df = pd.read_csv("generated/dbscan_results.csv", encoding="utf-8-sig")
if_df = pd.read_csv("generated/isolation_forest_results.csv", encoding="utf-8-sig")
Z = ["z_age", "z_income", "z_balance", "z_product_count"]
data = (df.merge(km_df, on="customer_id")
          .merge(db_df, on="customer_id")
          .merge(if_df[["customer_id", "is_anomaly", "anomaly_score"]], on="customer_id"))
Xz = data[Z].to_numpy()
km_lab = data["cluster"].to_numpy()
db_lab = data["dbscan_cluster"].to_numpy()
if_lab = data["is_anomaly"].to_numpy()
n_if = int(if_lab.sum())
RAW_FEATURES = ["age", "income", "balance", "product_count"]
if_means = data.loc[if_lab == 1, RAW_FEATURES].mean()
all_means = data[RAW_FEATURES].mean()

print("=" * 74)
print("一、两模型基本信息")
print("=" * 74)
km_k = len(set(km_lab))
db_k = len(set(db_lab)) - (1 if -1 in db_lab else 0)
noise_mask = db_lab == -1
n_noise = int(noise_mask.sum())

sil_km = silhouette_score(Xz, km_lab)
db_keep = ~noise_mask
sil_db = silhouette_score(Xz[db_keep], db_lab[db_keep]) if db_k >= 2 else np.nan
ari = adjusted_rand_score(km_lab, db_lab)

km_sizes = pd.Series(km_lab).value_counts().sort_index()
db_sizes = pd.Series(db_lab[db_keep]).value_counts().sort_index()

print(f"KMeans：K={km_k}，轮廓系数={sil_km:.6f}（全部 {len(data)} 个样本）")
print(f"  各簇人数：{km_sizes.to_dict()}")
print(f"DBSCAN：簇数={db_k}，噪声={n_noise}（{n_noise / len(data) * 100:.2f}%），"
      f"轮廓系数={sil_db:.6f}（仅 {int(db_keep.sum())} 个非噪声样本）")
print(f"  各簇人数：{db_sizes.to_dict()}")
print(f"\n两模型划分一致性 ARI = {ari:.6f}")
print("说明：ARI 把 DBSCAN 的 -1 当作独立标签参与计算。")

overlap = int(((if_lab == 1) & (db_lab == -1)).sum())
print(f"\nIsolation Forest：异常 {n_if} 人（{n_if / len(data) * 100:.2f}%），"
      f"contamination=0.05，random_state=42")
print(f"IF 异常与 DBSCAN 噪声的重合人数：{overlap} / {n_if}"
      f"（重合率 {overlap / n_if * 100:.2f}%）")
print("注意：Isolation Forest 是异常检测，KMeans/DBSCAN 是聚类分群，任务性质不同。")

print("\n" + "=" * 74)
print("二、簇规模比较")
print("=" * 74)
for name, sizes in [("KMeans", km_sizes), ("DBSCAN", db_sizes)]:
    print(f"{name:<7} 簇数={len(sizes)}  最大={int(sizes.max())}  最小={int(sizes.min())}  "
          f"均值={sizes.mean():.1f}  最大/最小={sizes.max() / sizes.min():.2f}")
print(f"KMeans 覆盖率：100.00%")
print(f"DBSCAN 覆盖率：{db_keep.mean() * 100:.2f}%（噪声不归簇）")

# 交叉表：KMeans 簇 × DBSCAN 标签
ct = pd.crosstab(pd.Series(km_lab, name="KMeans簇"),
                 pd.Series(db_lab, name="DBSCAN标签"))
print("\n交叉表（列 -1 为噪声）：")
print(ct.to_string())

# ==================== 保存 CSV ====================
rows = [
    {"模型": "KMeans", "任务类型": "聚类（客户分群）",
     "参数": "K=4, random_state=42, n_init=10", "聚类数量": km_k,
     "噪声/异常点数量": 0, "噪声/异常占比(%)": 0.0, "样本覆盖率(%)": 100.0,
     "轮廓系数": round(float(sil_km), 6), "轮廓系数口径": "全部样本",
     "最大簇人数": int(km_sizes.max()), "最小簇人数": int(km_sizes.min()),
     "各簇人数": str(km_sizes.tolist()),
     "适用问题": "把全部客户划分为 K 个互斥客群，用于分层运营"},
    {"模型": "DBSCAN", "任务类型": "聚类（密度聚类，对照实验）",
     "参数": "eps=1.0, min_samples=7", "聚类数量": db_k,
     "噪声/异常点数量": n_noise, "噪声/异常占比(%)": round(n_noise / len(data) * 100, 2),
     "样本覆盖率(%)": round(float(db_keep.mean() * 100), 2),
     "轮廓系数": round(float(sil_db), 6) if not np.isnan(sil_db) else np.nan,
     "轮廓系数口径": "非噪声样本（285 个）",
     "最大簇人数": int(db_sizes.max()), "最小簇人数": int(db_sizes.min()),
     "各簇人数": str(db_sizes.tolist()),
     "适用问题": "不指定簇数的密度分群，同时标记低密度噪声点"},
    {"模型": "Isolation Forest", "任务类型": "异常检测",
     "参数": "n_estimators=200, contamination=0.05, random_state=42",
     "聚类数量": "不适用", "噪声/异常点数量": n_if,
     "噪声/异常占比(%)": round(n_if / len(data) * 100, 2), "样本覆盖率(%)": 100.0,
     "轮廓系数": "不适用（非聚类任务）", "轮廓系数口径": "—",
     "最大簇人数": "—", "最小簇人数": "—", "各簇人数": "—",
     "适用问题": "识别特征组合偏离主体的个体，用于人工复核名单"},
]
pd.DataFrame(rows).to_csv("generated/model_comparison.csv", index=False, encoding="utf-8-sig")

# ==================== 绘图 ====================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.9))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

ax = axes[0]
for c in sorted(set(km_lab)):
    m = km_lab == c
    ax.scatter(data.loc[m, "income"], data.loc[m, "balance"], s=34, alpha=0.7,
               color=colors[c % len(colors)], edgecolors="white", linewidths=0.4,
               label=f"簇 {c}（{int(m.sum())} 人）")
ax.set_title(f"KMeans（K={km_k}，覆盖 100%）", fontsize=12)
ax.set_xlabel("income（万元）", fontsize=10)
ax.set_ylabel("balance（万元）", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

ax = axes[1]
for c in sorted(set(db_lab)):
    m = db_lab == c
    if c == -1:
        ax.scatter(data.loc[m, "income"], data.loc[m, "balance"], s=44, c="black",
                   marker="x", label=f"噪声（{int(m.sum())} 人）")
    else:
        ax.scatter(data.loc[m, "income"], data.loc[m, "balance"], s=34, alpha=0.7,
                   color=colors[c % len(colors)], edgecolors="white", linewidths=0.4,
                   label=f"簇 {c}（{int(m.sum())} 人）")
ax.set_title(f"DBSCAN（{db_k} 簇 + {n_noise} 噪声）", fontsize=12)
ax.set_xlabel("income（万元）", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

ax = axes[2]
metrics = ["轮廓系数", "样本覆盖率(%)", "噪声占比(%)"]
km_vals = [sil_km, 100.0, 0.0]
db_vals = [sil_db, float(db_keep.mean() * 100), n_noise / len(data) * 100]
x = np.arange(len(metrics))
ax.bar(x - 0.19, km_vals, width=0.36, color="#4C72B0", label="KMeans")
ax.bar(x + 0.19, db_vals, width=0.36, color="#DD8452", label="DBSCAN")
for i, (a, b) in enumerate(zip(km_vals, db_vals)):
    ax.text(x[i] - 0.19, a + 1.5, f"{a:.3f}" if i == 0 else f"{a:.1f}", ha="center", fontsize=9)
    ax.text(x[i] + 0.19, b + 1.5, f"{b:.3f}" if i == 0 else f"{b:.1f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 115)
ax.set_title("关键指标对比（注意口径不同）", fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.suptitle(f"HW2 · 模型评估：KMeans vs DBSCAN（ARI = {ari:.3f}）", fontsize=15)
plt.tight_layout()
plt.savefig("outputs/03_model_comparison.png", dpi=150)
plt.close()

# ==================== 报告 ====================
md = f"""# HW2 · 模型评估与比较报告

> 同一份预处理数据（300 位客户、四个标准化特征、StandardScaler），
> 数据为课程**模拟客户数据**，非真实银行业务数据。

## 一、三类任务总表

| 任务类型 | 模型 | 参数 | 输出 | 本次结果 | 主要指标 |
|---|---|---|---|---|---|
| 聚类（客户分群） | KMeans | K={km_k}, random_state=42, n_init=10 | 每客户一个簇号 | {km_k} 簇：{km_sizes.tolist()} | 轮廓系数 {sil_km:.6f}（全部 {len(data)} 样本） |
| 聚类（密度聚类，对照） | DBSCAN | eps=1.0, min_samples=7 | 簇号或噪声 -1 | {db_k} 簇 + {n_noise} 噪声（{n_noise / len(data) * 100:.2f}%） | 轮廓系数 {sil_db:.6f}（仅 285 非噪声） |
| 异常检测 | Isolation Forest | n_estimators=200, contamination=0.05, random_state=42 | 0/1 标记 + 异常分数 | 异常 **{n_if}** 人（{n_if / len(data) * 100:.2f}%） | 异常数/占比、异常画像差异；轮廓系数不适用 |

**不可同榜排名**：前两者是聚类，目标是把客户分成互斥群体；Isolation Forest 是异常检测，
目标是找出偏离主体的个体，输出不是簇号。三者的指标没有共同量纲，不能把
"轮廓系数 0.54 > 0.41"解读为 DBSCAN 优于 KMeans，也不能用异常比例高低给 Isolation Forest 排名。

**口径提醒**：DBSCAN 的轮廓系数是在剔除 {n_noise} 个噪声点后计算的，样本集合与 KMeans 不同。

## 二、簇规模

| 模型 | 各簇人数 | 最大 | 最小 | 最大/最小 |
|---|---|---|---|---|
| KMeans | {km_sizes.tolist()} | {int(km_sizes.max())} | {int(km_sizes.min())} | {km_sizes.max() / km_sizes.min():.2f} |
| DBSCAN | {db_sizes.tolist()} | {int(db_sizes.max())} | {int(db_sizes.min())} | {db_sizes.max() / db_sizes.min():.2f} |

KMeans 四个簇规模较均衡（50~102 人）；DBSCAN 只有 2 个簇，且 242 人的大簇把中等与低值客户合并，
业务分辨率明显更粗。

## 三、两模型划分一致性

- ARI = **{ari:.6f}**（DBSCAN 的噪声作为 -1 标签参与计算）。
- 交叉表：

```
{ct.to_string()}
```

ARI 不高，说明二者不是同一划分的不同编号，而是结构层面不同：
KMeans 强行把所有客户切成 4 份，DBSCAN 把密度连续的主体并成 2 团、把稀疏的 15 个点标为噪声。

## 四、Isolation Forest 结果（异常检测）

- 参数：`n_estimators=200, contamination=0.05, random_state=42`；特征与聚类完全相同。
- 异常客户：**{n_if} 人（{n_if / len(data) * 100:.2f}%）**，正常 {len(data) - n_if} 人。
- 异常客户均值：年龄 {if_means['age']:.2f} 岁、收入 {if_means['income']:.2f} 万元、
  余额 {if_means['balance']:.2f} 万元、产品数 {if_means['product_count']:.2f} 个；
  整体均值：年龄 {all_means['age']:.2f}、收入 {all_means['income']:.2f}、
  余额 {all_means['balance']:.2f}、产品数 {all_means['product_count']:.2f}。
- 异常客户整体偏"高龄、高收入、高余额、多产品"，属于偏离主群体的少数客户，
  应做人工复核与差异化服务，**不能认定为欺诈**。
- 与 DBSCAN 噪声的重合：**{overlap} / {n_if}**（{overlap / n_if * 100:.2f}%）。
  两种方法都指向"偏离主体"的客户，但判定机制不同（密度可达 vs 随机切分隔离难度），
  名单不完全一致，业务上可互为补充而非互相替代。

## 五、优缺点分析

| 维度 | KMeans | DBSCAN |
|---|---|---|
| 优点 | 覆盖全部客户；簇数可控；结果稳定（前序实验 ARI=1）；规模均衡，便于分层运营 | 不需要指定簇数；能识别噪声/异常客户；对非凸结构更稳健 |
| 局限 | 假设凸球形簇；对异常值与量纲敏感（需标准化）；K 需人为选择 | 高维下密度退化；参数 eps/min_samples 敏感；簇数被压成 2 个；15 个客户无归属 |
| 本次表现 | 4 簇 50/54/102/94，轮廓系数 0.412272 | 2 簇 242/43 + 15 噪声，非噪声轮廓系数 0.541996 |

Isolation Forest 的定位不同：它不做分群，只回答"哪些个体不像主体人群"；
优点是无需指定簇数、可给异常分数排序，局限是 contamination 为人为设定、异常不等于风险。

## 六、哪个模型更适合当前客户数据

**主分群选 KMeans，DBSCAN 作为密度聚类对照，Isolation Forest 作为独立的异常检测任务。**

理由（不只依据单一指标）：

1. **业务可用性**：零售营销需要覆盖全部客户并做 4 类差异化策略；KMeans 覆盖率 100%、
   四簇规模均衡、画像可解释。DBSCAN 的 242 人超级簇无法支撑分层运营。
2. **指标口径**：DBSCAN 的 0.542 是在剔除 15 个难分点、只剩 2 簇的情况下得到的，
   簇数更少本就更易获得高轮廓系数，不能据此认定其划分更优。
3. **噪声的业务价值**：DBSCAN 找出的 15 个噪声客户平均年龄 43.93 岁、收入 30.85 万元、
   余额 31.87 万元，明显高于整体，属于"高价值但特征偏离主体"的客户，
   可作为人工复核/专属维护名单，而不是丢弃。
4. **稳定性**：前序实验已证明 KMeans 对随机种子不敏感；DBSCAN 对 eps 敏感（0.9~1.1 结果尚稳，
   超过 1.2 后迅速退化为 1 簇）。
5. **任务互补**：聚类负责"把客户分成几群"，Isolation Forest 负责"挑出不像主体的个体"，
   两者回答不同问题；完整的数据挖掘流程同时需要这两类结果，而不是二选一。

## 七、输出

- `generated/model_comparison.csv`（三行：KMeans / DBSCAN / Isolation Forest）
- `outputs/03_model_comparison.png`（KMeans、DBSCAN 散点 + 两项聚类指标；异常检测单列于
  `outputs/03_isolation_forest.png`，避免混用指标）
"""
with open("generated/model_comparison.md", "w", encoding="utf-8") as f:
    f.write(md)
print("\n已保存：generated/model_comparison.csv")
print("已保存：generated/model_comparison.md")
print("已保存：outputs/03_model_comparison.png")
