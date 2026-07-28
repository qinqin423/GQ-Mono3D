import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

path = "q_pred_dump.jsonl"

qs = []
scores = []

with open(path, "r") as f:
    for line in f:
        try:
            r = json.loads(line)
        except:
            continue

        if "geo_quality" not in r:
            continue

        if "score" not in r:
            continue

        qs.append(float(r["geo_quality"]))
        scores.append(float(r["score"]))

q = np.array(qs)
score = np.array(scores)

print("===== Statistics =====")
print("samples =", len(q))
print("Q mean =", q.mean())
print("score mean =", score.mean())

pearson = pearsonr(q, score)[0]
spearman = spearmanr(q, score)[0]

print("Pearson =", pearson)
print("Spearman =", spearman)

# scatter
np.random.seed(0)
idx = np.random.choice(
    len(q),
    min(5000, len(q)),
    replace=False
)

plt.figure(figsize=(6,5))
plt.scatter(
    q[idx],
    score[idx],
    s=5,
    alpha=0.25
)

plt.xlabel("Geometry Reliability Score Q")
plt.ylabel("Detection Confidence Score")
plt.title("Q-Score Correlation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("q_score_scatter.png", dpi=300)

# bin analysis
bins = [0.45,0.50,0.55,0.60,0.65,0.70,0.75]

labels = []
means = []

for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (q>=lo) & (q<hi)

    if mask.sum()==0:
        continue

    labels.append(f"{lo:.2f}-{hi:.2f}")
    means.append(score[mask].mean())

plt.figure(figsize=(7,4))
plt.bar(labels, means)

plt.xlabel("Q Range")
plt.ylabel("Mean Score")
plt.title("Mean Detection Score under Different Q Ranges")
plt.tight_layout()

plt.savefig(
    "q_score_bin_bar.png",
    dpi=300
)

print("[saved] q_score_scatter.png")
print("[saved] q_score_bin_bar.png")
