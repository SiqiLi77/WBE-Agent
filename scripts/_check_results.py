"""查看 Agent 调查结果的快速分析脚本。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd

# ── 加载数据 ──
results_path = "outputs/investigation_results.csv"
labels_path = "data/labeled/auto_labeled_events.csv"

df = pd.read_csv(results_path)
labels = pd.read_csv(labels_path)

print(f"Agent 已完成: {len(df)} / 152 个事件\n")

# ── 1. Agent 分类分布 ──
print("=" * 60)
print("1. Agent 分类分布")
print("=" * 60)
print(df["classification"].value_counts().to_string())

print(f"\n各分类平均置信度:")
print(df.groupby("classification")["confidence"].mean().round(2).to_string())

# ── 2. Token 消耗 ──
avg_tokens = df["total_tokens"].mean()
total_tokens = df["total_tokens"].sum()
avg_tools = df["tool_calls_count"].mean()
print(f"\n{'=' * 60}")
print(f"2. 资源消耗")
print(f"{'=' * 60}")
print(f"每事件平均 Token: {avg_tokens:.0f}")
print(f"总 Token 消耗: {total_tokens:,}")
print(f"每事件平均工具调用: {avg_tools:.1f} 次")

# ── 3. 各州分类分布 ──
def get_state(sid):
    parts = sid.split("_")
    if parts[0] == "CDC":
        return parts[2].upper()
    return parts[1].upper()

df["state"] = df["site_id"].apply(get_state)
print(f"\n{'=' * 60}")
print(f"3. 各州 Agent 分类分布")
print(f"{'=' * 60}")
ct = pd.crosstab(df["state"], df["classification"])
print(ct.to_string())


# ── 4. Agent vs 自动标注对比 ──
print(f"\n{'=' * 60}")
print(f"4. Agent 分类 vs 自动标注 对比")
print(f"{'=' * 60}")

merged = df.merge(labels[["event_id", "auto_label", "auto_confidence"]], on="event_id", how="inner")
print(f"可对比事件数: {len(merged)}")

# 混淆矩阵
print(f"\n混淆矩阵 (行=自动标注, 列=Agent分类):")
confusion = pd.crosstab(merged["auto_label"], merged["classification"], margins=True)
print(confusion.to_string())

# 一致率
agree = (merged["auto_label"] == merged["classification"]).sum()
print(f"\n完全一致: {agree}/{len(merged)} ({agree/len(merged)*100:.1f}%)")

# 各类别一致率
print(f"\n各自动标注类别 → Agent 分类去向:")
for label in sorted(merged["auto_label"].unique()):
    subset = merged[merged["auto_label"] == label]
    agent_dist = subset["classification"].value_counts()
    print(f"\n  [{label}] (n={len(subset)}):")
    for cls, cnt in agent_dist.items():
        print(f"    → {cls}: {cnt} ({cnt/len(subset)*100:.0f}%)")

# ── 5. 关键发现：Agent 改变了哪些判断 ──
print(f"\n{'=' * 60}")
print(f"5. Agent 改变了自动标注判断的案例")
print(f"{'=' * 60}")

changed = merged[merged["auto_label"] != merged["classification"]]
print(f"Agent 改变判断: {len(changed)}/{len(merged)} ({len(changed)/len(merged)*100:.1f}%)")

# 最有趣的变化：uncertain → 具体分类
uncertain_resolved = changed[changed["auto_label"] == "uncertain"]
print(f"\n自动标注 uncertain → Agent 给出具体分类: {len(uncertain_resolved)} 个")
if len(uncertain_resolved) > 0:
    resolved_dist = uncertain_resolved["classification"].value_counts()
    for cls, cnt in resolved_dist.items():
        print(f"  → {cls}: {cnt}")

# ── 6. 典型案例展示 ──
print(f"\n{'=' * 60}")
print(f"6. 各分类典型案例")
print(f"{'=' * 60}")
for cls in ["epidemic", "environmental", "mixed", "sampling", "uncertain"]:
    subset = df[df["classification"] == cls]
    if len(subset) > 0:
        row = subset.iloc[0]
        print(f"\n--- {cls.upper()} (n={len(subset)}) ---")
        print(f"  案例: {row['event_id']}")
        print(f"  站点: {row['site_id']}")
        print(f"  日期: {row['anomaly_date']}")
        print(f"  置信度: {row['confidence']}")
        print(f"  摘要: {row['summary'][:200]}")

print(f"\n{'=' * 60}")
print("分析完成")
print(f"{'=' * 60}")
