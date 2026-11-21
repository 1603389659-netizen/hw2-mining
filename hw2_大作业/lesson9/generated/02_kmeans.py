# -*- coding: utf-8 -*-
"""
HW2 第 2 步：KMeans 客户聚类

- 读取预处理结果 generated/cleaned_data.csv，使用 4 个标准化特征（z_ 前缀）
- K = 2..9 逐一计算 inertia_ 与轮廓系数
- 肘部法 + 轮廓系数共同选 K：在 K>=3 中，取轮廓系数不低于最优值 98% 的最小 K（简约原则），
  并与肘部拐点交叉验证
- 输出各簇人数、占比、画像，保存结果 CSV、报告与中文图
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
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
OUT_CSV = "generated/kmeans_results.csv"
OUT_MD = "generated/kmeans_report.md"
OUT_PNG = "outputs/01_kmeans.png"
os.makedirs("outputs", exist_ok=True)

Z = ["z_age", "z_income", "z_balance", "z_product_count"]
df = pd.read_csv(CLEANED, encoding="utf-8-sig")
Xz = df[Z].to_numpy()
print("=" * 72)
print("一、输入数据")
print("=" * 72)
print(f"{CLEANED}：{df.shape[0]} 行 × {df.shape[1]} 列")
print(f"模型特征：{Z}（已标准化，不含 customer_id / gender / city）")

# ==================== 遍历 K ====================
print("\n" + "=" * 72)
print("二、K = 2..9：inertia_ 与轮廓系数")
print("=" * 72)
rows = []
models = {}
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xz)
    s = silhouette_score(Xz, labels)
    sizes = np.bincount(labels).tolist()
    models[k] = (km, labels)
    rows.append({"K": k, "inertia": round(float(km.inertia_), 4),
                 "轮廓系数": round(float(s), 6), "最小簇人数": int(min(sizes))})
    print(f"K={k}：inertia={km.inertia_:>9.4f}  轮廓系数={s:.6f}  "
          f"各簇人数={sizes}")

res = pd.DataFrame(rows)

# ==================== 选 K ====================
print("\n" + "=" * 72)
print("三、选 K")
print("=" * 72)
# 肘部：K=2..9 曲线上点到首尾连线距离最大处（两轴先归一化，避免量纲影响）
ks = res["K"].to_numpy().astype(float)
sse = res["inertia"].to_numpy().astype(float)
kx = (ks - ks.min()) / (ks.max() - ks.min())
sy = (sse - sse.min()) / (sse.max() - sse.min())
p1, p2 = np.array([kx[0], sy[0]]), np.array([kx[-1], sy[-1]])
vec = p2 - p1
pts = np.column_stack([kx, sy]) - p1
# 二维叉积的模 / 底边长度 = 点到直线距离
dist = np.abs(vec[0] * pts[:, 1] - vec[1] * pts[:, 0]) / np.linalg.norm(vec)
elbow_k = int(ks[int(np.argmax(dist))])

sub = res[res["K"] >= 3]
best_sil = float(sub["轮廓系数"].max())
best_k_sil = int(sub.loc[sub["轮廓系数"].idxmax(), "K"])
# 简约原则：轮廓系数 >= 最优值 98% 的最小 K
candidates = sub[sub["轮廓系数"] >= 0.98 * best_sil]["K"].tolist()
chosen_k = int(min(candidates))

print(f"几何肘部 K = {elbow_k}")
print(f"K>=3 中轮廓系数最优：K={best_k_sil}（{best_sil:.6f}）")
print(f"轮廓系数 ≥ 98%×最优值的 K：{candidates} → 取最小 K = {chosen_k}")
print(f"最终选择 K = {chosen_k}"
      f"（{'与肘部一致' if elbow_k == chosen_k else '与肘部相差 ' + str(abs(elbow_k - chosen_k)) + '，以轮廓系数+简约原则为准'}）")

km, labels = models[chosen_k]
df["cluster"] = labels
N = len(df)

# ==================== 各簇人数与画像 ====================
print("\n" + "=" * 72)
print(f"四、K={chosen_k} 聚类结果")
print("=" * 72)
g = df.groupby("cluster")
prof = pd.DataFrame({
    "人数": g.size(),
    "占比(%)": (g.size() / N * 100).round(2),
    "平均年龄(岁)": g["age"].mean().round(2),
    "平均收入(万元)": g["income"].mean().round(2),
    "平均余额(万元)": g["balance"].mean().round(2),
    "平均产品数(个)": g["product_count"].mean().round(2),
})
print(prof.to_string())
print(f"\n人数合计：{int(prof['人数'].sum())}（应为 {N}）")
print(f"inertia_ = {km.inertia_:.4f}")
print(f"整体轮廓系数 = {silhouette_score(Xz, labels):.6f}")

df[["customer_id", "cluster"]].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存：{OUT_CSV}")

# ==================== 绘图 ====================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

ax = axes[0]
ax.plot(res["K"], res["inertia"], "o-", color="#4C72B0", linewidth=2)
ax.axvline(elbow_k, color="#C44E52", linestyle="--", alpha=0.8,
           label=f"几何肘部 K={elbow_k}")
ax.set_title("肘部法：SSE 随 K 的下降", fontsize=13)
ax.set_xlabel("簇数 K", fontsize=11)
ax.set_ylabel("inertia_（簇内平方和）", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(res["K"], res["轮廓系数"], "o-", color="#55A868", linewidth=2)
ax.axvline(chosen_k, color="#C44E52", linestyle="--", alpha=0.8,
           label=f"选定 K={chosen_k}")
ax.axhline(best_sil, color="gray", linestyle=":", alpha=0.7,
           label=f"K>=3 最优 {best_sil:.3f}")
ax.set_title("轮廓系数随 K 的变化", fontsize=13)
ax.set_xlabel("簇数 K", fontsize=11)
ax.set_ylabel("轮廓系数", fontsize=11)
ax.set_ylim(0.3, 0.55)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

ax = axes[2]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
for c in sorted(df["cluster"].unique()):
    sub_c = df[df["cluster"] == c]
    ax.scatter(sub_c["income"], sub_c["balance"], s=40, alpha=0.72,
               color=colors[c % len(colors)], edgecolors="white", linewidths=0.4,
               label=f"簇 {c}（{len(sub_c)} 人）")
    ax.scatter(sub_c["income"].mean(), sub_c["balance"].mean(), marker="*", s=560,
               color=colors[c % len(colors)], edgecolors="black", linewidths=1.1, zorder=5)
ax.set_title(f"KMeans 聚类结果（K={chosen_k}，星形为簇中心）", fontsize=13)
ax.set_xlabel("年收入 income（万元）", fontsize=11)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

fig.suptitle("HW2 · KMeans 客户聚类（标准化四特征）", fontsize=15)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"已保存：{OUT_PNG}")

# ==================== 报告 ====================
md = f"""# HW2 · KMeans 客户聚类报告

> 数据：`{CLEANED}`（源自 `data/mall_customers_sim.csv` 的**模拟客户数据**，非真实银行业务数据）。
> 特征：`age`、`income`、`balance`、`product_count` 的 StandardScaler 标准化值（z_ 前缀）。
> `customer_id`、`gender`、`city` 未进入模型。

## 一、K 的选择

参数固定：`KMeans(n_clusters=K, random_state=42, n_init=10)`，K = 2..9。

| K | inertia_ | 轮廓系数 | 最小簇人数 |
|---|---|---|---|
"""
for _, r in res.iterrows():
    md += f"| {int(r['K'])} | {r['inertia']:.4f} | {r['轮廓系数']:.6f} | {int(r['最小簇人数'])} |\n"

md += f"""
- **肘部法**（K=2..9 曲线上点到首尾连线距离最大）：**K = {elbow_k}**。
- **轮廓系数**：K>=3 时 K={best_k_sil} 最高（{best_sil:.6f}）。
- **选 K 规则**：K=2 虽然轮廓系数更高（{float(res.iloc[0]['轮廓系数']):.6f}），但二分会把不同客群硬合并，
  业务分辨率不足；因此在 K>=3 中取"轮廓系数不低于最优值 98%"的最小 K，即 **K = {chosen_k}**。
- 该 K 与几何肘部 {'一致' if elbow_k == chosen_k else f'相差 {abs(elbow_k - chosen_k)}'}：
  K={chosen_k} 时 inertia 的边际下降仍明显，之后趋缓，轮廓系数也处于高位。

## 二、各簇人数与占比（K={chosen_k}）

| 簇 | 人数 | 占比 |
|---|---|---|
"""
for c, r in prof.iterrows():
    md += f"| {int(c)} | {int(r['人数'])} | {r['占比(%)']:.2f}% |\n"

md += f"""
合计 **{int(prof['人数'].sum())}** 人，与样本量 {N} 一致。
整体轮廓系数 **{silhouette_score(Xz, labels):.6f}**，inertia_ = **{km.inertia_:.4f}**。

## 三、各簇画像（真实统计值）

| 簇 | 人数 | 平均年龄 | 平均收入(万元) | 平均余额(万元) | 平均产品数 |
|---|---|---|---|---|---|
"""
for c, r in prof.iterrows():
    md += (f"| {int(c)} | {int(r['人数'])} | {r['平均年龄(岁)']:.2f} | "
           f"{r['平均收入(万元)']:.2f} | {r['平均余额(万元)']:.2f} | "
           f"{r['平均产品数(个)']:.2f} |\n")

gm = df[["age", "income", "balance", "product_count"]].mean()
md += f"""
全局均值参考：年龄 {gm['age']:.2f} 岁、收入 {gm['income']:.2f} 万元、
余额 {gm['balance']:.2f} 万元、产品数 {gm['product_count']:.2f} 个。

各簇相对全局的高低关系（用于后续命名，不在此预设业务名称）：

"""
for c, r in prof.iterrows():
    rel = []
    rel.append("年龄偏高" if r["平均年龄(岁)"] > gm["age"] else "年龄偏低")
    rel.append("收入偏高" if r["平均收入(万元)"] > gm["income"] else "收入偏低")
    rel.append("余额偏高" if r["平均余额(万元)"] > gm["balance"] else "余额偏低")
    rel.append("产品数偏多" if r["平均产品数(个)"] > gm["product_count"] else "产品数偏少")
    md += f"- 簇 {int(c)}（{int(r['人数'])} 人）：{'、'.join(rel)}。\n"

md += f"""
## 四、输出文件

- `{OUT_CSV}`：customer_id 与 cluster 对应表（300 行）。
- `{OUT_PNG}`：左 SSE 曲线 + 中轮廓系数 + 右 income×balance 散点（星形为簇中心，中文图例）。

## 五、说明

- 聚类使用标准化后的四特征；散点图只展示 income × balance 两个维度，
  age 与 product_count 的区分信息在二维图上不可见。
- 簇编号不代表业务含义，业务名称在 `05_business_analysis.py` 中依据真实均值生成。
"""
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)
print(f"已保存：{OUT_MD}")
