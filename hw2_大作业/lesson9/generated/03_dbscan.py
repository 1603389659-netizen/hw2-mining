# -*- coding: utf-8 -*-
"""
HW2 第 3 步：DBSCAN 客户聚类

- 输入与 KMeans 完全相同的标准化四特征
- 网格尝试 eps 与 min_samples，记录簇数、噪声数、噪声占比、轮廓系数（不含噪声）
- 选择规则：簇数 2~6、噪声占比 <= 15% 的参数组合中，轮廓系数最高；
  若无满足条件的组合，则如实报告并取覆盖最好的组合（不伪造"成功"结果）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

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
OUT_CSV = "generated/dbscan_results.csv"
OUT_MD = "generated/dbscan_report.md"
OUT_PNG = "outputs/02_dbscan.png"
os.makedirs("outputs", exist_ok=True)

Z = ["z_age", "z_income", "z_balance", "z_product_count"]
df = pd.read_csv(CLEANED, encoding="utf-8-sig")
Xz = df[Z].to_numpy()
print("=" * 72)
print("一、输入数据")
print("=" * 72)
print(f"{CLEANED}：{df.shape[0]} 行，特征 {Z}（与 KMeans 完全相同）")

# ==================== 网格搜索 ====================
print("\n" + "=" * 72)
print("二、参数网格搜索（eps 0.8~2.2，min_samples 3~10）")
print("=" * 72)
rows = []
for eps in np.round(np.arange(0.8, 2.21, 0.1), 2):
    for ms in range(3, 11):
        db = DBSCAN(eps=float(eps), min_samples=ms)
        lab = db.fit_predict(Xz)
        n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
        n_noise = int((lab == -1).sum())
        sil = np.nan
        if n_clusters >= 2 and n_noise < len(lab):
            mask = lab != -1
            if len(set(lab[mask])) >= 2:
                sil = silhouette_score(Xz[mask], lab[mask])
        sizes = sorted([int((lab == c).sum()) for c in set(lab) if c != -1], reverse=True)
        rows.append({"eps": float(eps), "min_samples": ms, "簇数": n_clusters,
                     "噪声数": n_noise, "噪声占比(%)": round(n_noise / len(lab) * 100, 2),
                     "轮廓系数(非噪声)": sil, "各簇人数": str(sizes)})

grid = pd.DataFrame(rows)
valid = grid[(grid["簇数"] >= 2) & (grid["簇数"] <= 6) &
             (grid["噪声占比(%)"] <= 15) & grid["轮廓系数(非噪声)"].notna()]

print("满足条件（2≤簇数≤6，噪声≤15%）的组合数：", len(valid))
if len(valid):
    top = valid.sort_values("轮廓系数(非噪声)", ascending=False).head(10)
    print("\n前 10 个候选参数组合：")
    print(top[["eps", "min_samples", "簇数", "噪声数", "噪声占比(%)",
               "轮廓系数(非噪声)", "各簇人数"]].to_string(index=False))
    best = top.iloc[0]
    fallback = False
else:
    # 回退：覆盖最多且至少 2 簇的组合
    pool = grid[(grid["簇数"] >= 2) & grid["轮廓系数(非噪声)"].notna()]
    best = pool.sort_values(["噪声占比(%)", "轮廓系数(非噪声)"],
                           ascending=[True, False]).iloc[0]
    fallback = True
    print("\n没有组合满足预设条件，按噪声占比最低回退选择。")

eps_best = float(best["eps"])
ms_best = int(best["min_samples"])
print(f"\n最终参数：eps={eps_best}，min_samples={ms_best}")

# ==================== 最终模型 ====================
db = DBSCAN(eps=eps_best, min_samples=ms_best)
labels = db.fit_predict(Xz)
df["dbscan_cluster"] = labels
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())

print("\n" + "=" * 72)
print("三、DBSCAN 聚类结果")
print("=" * 72)
print(f"簇数：{n_clusters}    噪声点：{n_noise}（{n_noise / len(labels) * 100:.2f}%）")
print("各簇人数（-1 为噪声）：")
for c in sorted(set(labels)):
    m = labels == c
    tag = "噪声" if c == -1 else f"簇 {c}"
    print(f"  {tag:<5}：{int(m.sum()):>3d} 人，平均年龄 {df.loc[m, 'age'].mean():.2f}、"
          f"收入 {df.loc[m, 'income'].mean():.2f}、余额 {df.loc[m, 'balance'].mean():.2f}、"
          f"产品数 {df.loc[m, 'product_count'].mean():.2f}")

sil_db = np.nan
if n_clusters >= 2:
    mask = labels != -1
    sil_db = silhouette_score(Xz[mask], labels[mask])
    print(f"\n非噪声样本轮廓系数：{sil_db:.6f}（样本数 {int(mask.sum())}）")
else:
    print("\n簇数不足 2，无法计算轮廓系数。")

df[["customer_id", "dbscan_cluster"]].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"已保存：{OUT_CSV}")

grid.sort_values(["eps", "min_samples"]).to_csv(
    "generated/dbscan_grid.csv", index=False, encoding="utf-8-sig")
print("已保存：generated/dbscan_grid.csv（完整网格结果）")

# ==================== 绘图 ====================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

ax = axes[0]
for c in sorted(set(labels)):
    m = labels == c
    if c == -1:
        ax.scatter(df.loc[m, "income"], df.loc[m, "balance"], s=40, c="black",
                   marker="x", alpha=0.8, label=f"噪声（{int(m.sum())} 人）")
    else:
        ax.scatter(df.loc[m, "income"], df.loc[m, "balance"], s=40, alpha=0.72,
                   color=colors[c % len(colors)], edgecolors="white", linewidths=0.4,
                   label=f"簇 {c}（{int(m.sum())} 人）")
ax.set_title(f"DBSCAN 聚类（eps={eps_best}, min_samples={ms_best}）", fontsize=13)
ax.set_xlabel("年收入 income（万元）", fontsize=11)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

ax = axes[1]
sub = grid[(grid["min_samples"] == ms_best) & grid["轮廓系数(非噪声)"].notna()]
ax.plot(sub["eps"], sub["轮廓系数(非噪声)"], "o-", color="#55A868", linewidth=2)
ax.axvline(eps_best, color="#C44E52", linestyle="--", label=f"选定 eps={eps_best}")
ax.set_title(f"参数敏感性：固定 min_samples={ms_best}", fontsize=13)
ax.set_xlabel("eps", fontsize=11)
ax.set_ylabel("轮廓系数（非噪声样本）", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

fig.suptitle("HW2 · DBSCAN 客户密度聚类", fontsize=15)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"已保存：{OUT_PNG}")

# ==================== 报告 ====================
md = f"""# HW2 · DBSCAN 客户聚类报告

> 数据：`{CLEANED}`（课程**模拟客户数据**），特征与 KMeans 完全相同：
> `z_age`、`z_income`、`z_balance`、`z_product_count`。
> 算法：`DBSCAN(eps={eps_best}, min_samples={ms_best})`。

## 一、参数搜索过程

搜索范围：eps ∈ [0.8, 2.2]（步长 0.1），min_samples ∈ [3, 10]，共 {len(grid)} 种组合。
选择条件：簇数 2~6、噪声占比 ≤ 15%，在此范围内取非噪声样本轮廓系数最高者。
满足条件的组合共 **{len(valid)}** 个。

"""
if len(valid):
    md += """| eps | min_samples | 簇数 | 噪声数 | 噪声占比 | 轮廓系数(非噪声) | 各簇人数 |
|---|---|---|---|---|---|---|
"""
    for _, r in top.iterrows():
        md += (f"| {r['eps']} | {int(r['min_samples'])} | {int(r['簇数'])} | "
               f"{int(r['噪声数'])} | {r['噪声占比(%)']:.2f}% | "
               f"{r['轮廓系数(非噪声)']:.6f} | {r['各簇人数']} |\n")
else:
    md += "没有组合满足预设条件，按“噪声占比最低”回退选择。\n"

md += f"""
最终参数：**eps = {eps_best}，min_samples = {ms_best}**。

## 二、聚类结果

| 指标 | 数值 |
|---|---|
| 簇数 | {n_clusters} |
| 噪声点数 | {n_noise}（{n_noise / len(labels) * 100:.2f}%） |
| 非噪声样本轮廓系数 | {f'{sil_db:.6f}' if not np.isnan(sil_db) else '不适用（簇数<2）'} |

各簇人数与画像（-1 为噪声）：

| 标签 | 人数 | 平均年龄 | 平均收入(万元) | 平均余额(万元) | 平均产品数 |
|---|---|---|---|---|---|
"""
for c in sorted(set(labels)):
    m = labels == c
    md += (f"| {'噪声' if c == -1 else int(c)} | {int(m.sum())} | "
           f"{df.loc[m, 'age'].mean():.2f} | {df.loc[m, 'income'].mean():.2f} | "
           f"{df.loc[m, 'balance'].mean():.2f} | {df.loc[m, 'product_count'].mean():.2f} |\n")

md += f"""
## 三、DBSCAN 与 KMeans 的区别（机制层面）

| 对比项 | KMeans | DBSCAN |
|---|---|---|
| 簇形状 | 凸的、近似球形 | 任意形状（密度连通） |
| 簇数 | 必须预先指定 K | 由 eps/min_samples 自动决定 |
| 噪声处理 | 所有点都强制归簇 | 低密度点标记为噪声（-1） |
| 对异常值 | 敏感，会拉动质心 | 相对稳健，可识别为噪声 |
| 主要参数 | K、初始中心 | eps、min_samples |
| 结果覆盖 | 100% 样本 | 可能只覆盖部分样本 |

## 四、本次结果解读

- DBSCAN 在 eps={eps_best}、min_samples={ms_best} 下得到 **{n_clusters} 个簇**、
  噪声 **{n_noise}** 个（{n_noise / len(labels) * 100:.2f}%）。
- 若簇数与噪声占比显示 DBSCAN 把大部分客户并成少数大簇，说明四维标准化空间中样本密度连续、
  不存在明显的低密度间隔；这与 KMeans 能得到 {4} 个较均衡簇的结果形成对比。
- DBSCAN 的簇不保证覆盖全部客户，业务上需为噪声客户单独设计触达策略。

## 五、输出文件

- `{OUT_CSV}`：customer_id 与 dbscan_cluster（-1 为噪声）。
- `generated/dbscan_grid.csv`：{len(grid)} 组参数的完整搜索结果。
- `{OUT_PNG}`：左聚类散点（× 为噪声），右 eps 敏感性曲线。
"""
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)
print(f"已保存：{OUT_MD}")
