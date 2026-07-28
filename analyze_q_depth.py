import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

path = "q_pred_dump.jsonl"

qs = []
depths = []
scores = []

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue

        if "geo_quality" not in r or "depth" not in r:
            continue

        q = float(r["geo_quality"])
        depth = float(r["depth"])
        score = float(r.get("score", 0.0))

        if not np.isfinite(q) or not np.isfinite(depth):
            continue

        if depth <= 0:
            continue

        qs.append(q)
        depths.append(depth)
        scores.append(score)

q = np.array(qs)
depth = np.array(depths)
score = np.array(scores)

print("===== Q-Depth Analysis =====")
print(f"Num samples: {len(q)}")
print(f"Mean Q: {q.mean():.4f}")
print(f"Mean depth: {depth.mean():.4f}")
print(f"Pearson(Q, depth): {pearsonr(q, depth)[0]:.4f}")
print(f"Spearman(Q, depth): {spearmanr(q, depth)[0]:.4f}")

df = pd.DataFrame({
    "Q": q,
    "depth": depth,
    "score": score,
})
df.to_csv("q_depth_records.csv", index=False)

# sample scatter
np.random.seed(0)
n = min(5000, len(q))
idx = np.random.choice(len(q), n, replace=False)

plt.figure(figsize=(6, 5))
plt.scatter(q[idx], depth[idx], s=5, alpha=0.20)
plt.xlabel("Geometry Reliability Score Q")
plt.ylabel("Predicted Depth")
plt.title("Q-Depth Relationship")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("q_depth_scatter.png", dpi=300)

# bin analysis
bins = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
labels, mean_depths, counts = [], [], []

for lo, hi in zip(bins[:-1], bins[1:]):
    m = (q >= lo) & (q < hi)
    if m.sum() > 0:
        labels.append(f"{lo:.2f}-{hi:.2f}")
        mean_depths.append(depth[m].mean())
        counts.append(int(m.sum()))

plt.figure(figsize=(7, 4))
plt.bar(labels, mean_depths)
plt.xlabel("Q Range")
plt.ylabel("Mean Predicted Depth")
plt.title("Mean Depth under Different Q Ranges")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("q_depth_bin_bar.png", dpi=300)

pd.DataFrame({
    "Q_range": labels,
    "mean_depth": mean_depths,
    "count": counts,
}).to_csv("q_depth_bin_stats.csv", index=False)

print("[Saved] q_depth_records.csv")
print("[Saved] q_depth_scatter.png")
print("[Saved] q_depth_bin_bar.png")
print("[Saved] q_depth_bin_stats.csv")
