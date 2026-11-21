# -*- coding: utf-8 -*-
"""
HW2 第 5 步：业务分析（基于 KMeans K=4 的真实结果）

- 按簇统计人数、占比、年龄/收入/余额/产品数
- 依据各簇相对全局均值的高低自动生成中文业务名称
- 输出差异化建议、误分类与局限说明
- 保存 customer_segments.csv、business_analysis.md、outputs/04_customer_segments.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
km = pd.read_csv("generated/kmeans_results.csv", encoding="utf-8-sig")
df = df.merge(km, on="customer_id")
N = len(df)
FEATURES = ["age", "income", "balance", "product_count"]
g = df.groupby("cluster")

prof = pd.DataFrame({
    "人数": g.size(),
    "占比(%)": (g.size() / N * 100).round(2),
    "平均年龄": g["age"].mean().round(2),
    "平均收入": g["income"].mean().round(2),
    "平均余额": g["balance"].mean().round(2),
    "平均产品数": g["product_count"].mean().round(2),
})
gm = df[FEATURES].mean()
print("=" * 76)
print("一、KMeans（K=4）各簇真实统计")
print("=" * 76)
print(prof.to_string())
print(f"\n全局均值：年龄 {gm['age']:.2f}、收入 {gm['income']:.2f}、"
      f"余额 {gm['balance']:.2f}、产品数 {gm['product_count']:.2f}")


def name_segment(r):
    """依据相对全局均值的高低自动命名，不按簇编号硬编码。"""
    income_hi = r["平均收入"] > gm["income"]
    balance_hi = r["平均余额"] > gm["balance"]
    age_hi = r["平均年龄"] > gm["age"]
    prod_hi = r["平均产品数"] > gm["product_count"]
    if income_hi and balance_hi and r["平均收入"] > gm["income"] * 1.8:
        return "高净值中年客群" if age_hi else "高净值成长客群"
    if prod_hi and not income_hi:
        return "年轻多产品成长客群" if not age_hi else "多产品稳健客群"
    if income_hi and not balance_hi and age_hi:
        return "中年稳健理财客群"
    if not income_hi and not balance_hi:
        return "年轻低净值潜力客群" if not age_hi else "低净值待激活客群"
    return "中等价值综合客群"


prof["客群名称"] = [name_segment(prof.loc[c]) for c in prof.index]

print("\n" + "=" * 76)
print("二、客群命名（规则：与全局均值的相对高低）")
print("=" * 76)
for c in prof.index:
    r = prof.loc[c]
    print(f"簇 {c} → {r['客群名称']}（{int(r['人数'])} 人，"
          f"收入 {r['平均收入']:.2f}、余额 {r['平均余额']:.2f}、"
          f"年龄 {r['平均年龄']:.2f}、产品 {r['平均产品数']:.2f}）")

# 相对全局的 Z 值（用于描述与画图）
col_map = {"age": "平均年龄", "income": "平均收入",
           "balance": "平均余额", "product_count": "平均产品数"}
zprof = pd.DataFrame(index=prof.index)
for f in FEATURES:
    zprof[f] = (prof[col_map[f]] - gm[f]) / df[f].std(ddof=0)

# ==================== 业务建议（按实际特征生成） ====================
def suggestions(c, r):
    tips = []
    if r["平均收入"] > gm["income"] * 1.8:
        tips.append("配置专属客户经理与资产配置服务，重点做财富保值增值与流失预警")
        tips.append("可纳入私行/高端理财候选名单，但需逐一核实风险承受能力")
    elif r["平均产品数"] > gm["product_count"]:
        tips.append("以信用卡、消费信贷、定投等产品承接成长需求，提升主办行地位")
        tips.append("产品数已高于均值，避免重复推销，转向提升单产品使用深度")
    elif r["平均余额"] < gm["balance"] * 0.5:
        tips.append("采用低成本数字化触达，推送首卡、首贷与入门理财")
        tips.append("优先做客户激活与代发工资绑定，控制单客营销成本")
    else:
        tips.append("以存款留存和稳健理财转化为主，防止资金外流")
        tips.append("结合代发、缴费等场景提高账户活跃度与粘性")
    return tips

rows = []
for c in prof.index:
    r = prof.loc[c]
    seg = r["客群名称"]
    tips = suggestions(c, r)
    rows.append({
        "cluster": int(c), "客群名称": seg, "人数": int(r["人数"]),
        "占比(%)": float(r["占比(%)"]),
        "平均年龄(岁)": float(r["平均年龄"]), "平均收入(万元)": float(r["平均收入"]),
        "平均余额(万元)": float(r["平均余额"]), "平均产品数(个)": float(r["平均产品数"]),
        "画像摘要": (f"年龄{r['平均年龄']:.1f}岁，收入{r['平均收入']:.2f}万元，"
                    f"余额{r['平均余额']:.2f}万元，持有{r['平均产品数']:.2f}个产品"),
        "建议1": tips[0], "建议2": tips[1],
    })
seg_df = pd.DataFrame(rows)
seg_df.to_csv("generated/customer_segments.csv", index=False, encoding="utf-8-sig")
print("\n已保存：generated/customer_segments.csv")

# ==================== 绘图 ====================
fig, axes = plt.subplots(1, 3, figsize=(19.5, 6))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
labels_cn = [f"{int(c)}·{prof.loc[c, '客群名称']}" for c in prof.index]

ax = axes[0]
bars = ax.bar(range(len(prof)), prof["人数"], color=[colors[c % len(colors)] for c in prof.index])
for i, v in enumerate(prof["人数"]):
    ax.text(i, v + 2, f"{int(v)}人\n{prof['占比(%)'].iloc[i]:.1f}%", ha="center", fontsize=9)
ax.set_xticks(range(len(prof)))
ax.set_xticklabels(labels_cn, rotation=18, ha="right", fontsize=9)
ax.set_ylabel("客户人数", fontsize=11)
ax.set_title("各客群规模", fontsize=13)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
mat = zprof[FEATURES].to_numpy()
im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto", vmin=-2, vmax=2)
ax.set_xticks(range(4))
ax.set_xticklabels(["年龄", "收入", "余额", "产品数"], fontsize=10)
ax.set_yticks(range(len(prof)))
ax.set_yticklabels([f"簇 {c}" for c in prof.index], fontsize=10)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=10)
ax.set_title("客群特征相对全体的 Z 值", fontsize=13)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[2]
for c in prof.index:
    m = df["cluster"] == c
    ax.scatter(df.loc[m, "income"], df.loc[m, "balance"], s=34, alpha=0.7,
               color=colors[c % len(colors)], edgecolors="white", linewidths=0.4,
               label=f"{prof.loc[c, '客群名称']}（{int(prof.loc[c, '人数'])}人）")
    ax.scatter(prof.loc[c, "平均收入"], prof.loc[c, "平均余额"], marker="*", s=560,
               color=colors[c % len(colors)], edgecolors="black", linewidths=1.1, zorder=5)
ax.set_title("客群分布（收入 × 余额）", fontsize=13)
ax.set_xlabel("年收入（万元）", fontsize=11)
ax.set_ylabel("账户平均余额（万元）", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)

fig.suptitle("HW2 · 零售银行客户分群画像（模拟数据，KMeans K=4）", fontsize=15)
plt.tight_layout()
plt.savefig("outputs/04_customer_segments.png", dpi=150)
plt.close()
print("已保存：outputs/04_customer_segments.png")

# ==================== 报告 ====================
md = f"""# HW2 · 零售银行客户分群业务分析

> **重要声明**：本文基于课程提供的**模拟客户数据**（300 条）分析，结论仅用于方法演示，
> 不代表任何真实银行的客户结构或经营结果，不能直接作为经营决策依据。

## 一、分群总览

| 簇 | 客群名称 | 人数 | 占比 | 平均年龄 | 平均收入(万元) | 平均余额(万元) | 平均产品数 |
|---|---|---|---|---|---|---|---|
"""
for c in prof.index:
    r = prof.loc[c]
    md += (f"| {int(c)} | {r['客群名称']} | {int(r['人数'])} | {r['占比(%)']:.2f}% | "
           f"{r['平均年龄']:.2f} | {r['平均收入']:.2f} | {r['平均余额']:.2f} | "
           f"{r['平均产品数']:.2f} |\n")

md += f"""
全局均值：年龄 {gm['age']:.2f} 岁、收入 {gm['income']:.2f} 万元、
余额 {gm['balance']:.2f} 万元、产品数 {gm['product_count']:.2f} 个。

## 二、客群特征与业务建议

"""
for c in prof.index:
    r = prof.loc[c]
    tips = suggestions(c, r)
    md += f"""### {r['客群名称']}（簇 {int(c)}，{int(r['人数'])} 人，{r['占比(%)']:.2f}%）

- **特征**：平均年龄 {r['平均年龄']:.1f} 岁（全局 {gm['age']:.1f}），
  平均收入 {r['平均收入']:.2f} 万元（全局 {gm['income']:.2f}），
  平均余额 {r['平均余额']:.2f} 万元（全局 {gm['balance']:.2f}），
  平均持有 {r['平均产品数']:.2f} 个产品（全局 {gm['product_count']:.2f}）。
- **业务建议**：
  1. {tips[0]}；
  2. {tips[1]}。

"""

md += """## 三、差异化策略要点

1. **资源分配**：高净值客群人数最少但资产贡献最高，应配置高成本、高触达的服务；
   低净值客群人数最多，应以数字化、低成本方式覆盖，避免用人力做低效营销。
2. **产品策略**：产品数已高的客群不必继续叠加产品，转向提升使用深度与主账户地位；
   产品数低的客群优先做首卡、首贷与基础理财的"从 0 到 1"。
3. **留存优先**：收入高于均值但余额低于均值的客群存在资金外流，应优先做存款留存与代发绑定，
   而不是继续拉新。

## 四、可能的误分类与模型局限

1. **硬分配问题**：KMeans 把每个客户强制归入一个簇，处于两簇边界的客户（例如收入接近均值、
   余额居中）可能被分到相邻簇，个体层面的归属并不稳定。
2. **特征覆盖有限**：仅用年龄、收入、余额、产品数，缺少交易频次、渠道偏好、风险等级、
   持有产品类型等，画像维度不足。
3. **未使用分类字段**：gender、city 被排除在模型外，可能掩盖地域与性别差异；
   但这些字段涉及公平性与合规，不宜直接用于差异化定价。
4. **异常值影响**：数据中 50 行（16.67%）命中 IQR 异常，未删除，可能拉动簇中心；
   其中 15 个高价值离群客户被 DBSCAN 标为噪声，提示 KMeans 的划分对这些客户未必准确。
5. **无真值校验**：无监督评估只能衡量几何结构（轮廓系数 0.412272），不能证明业务有效性。
6. **样本量小**：300 条模拟数据，簇规模估计的置信区间较宽，不宜据此推断总体比例。

## 五、输出

- `generated/customer_segments.csv`：客群统计、画像摘要与建议。
- `outputs/04_customer_segments.png`：规模条形图 + 特征 Z 值热力图 + 收入×余额分布。
"""
with open("generated/business_analysis.md", "w", encoding="utf-8") as f:
    f.write(md)
print("已保存：generated/business_analysis.md")
