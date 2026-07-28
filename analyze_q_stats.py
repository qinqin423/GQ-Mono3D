import re
import numpy as np

log_path = "q_stats_stable_soft_gaqs.log"

means, stds, mins, maxs = [], [], [], []

pattern = re.compile(
    r"\[Q_STATS\]\s+mean=\s*([0-9.]+)\s+std=\s*([0-9.]+)\s+min=\s*([0-9.]+)\s+max=\s*([0-9.]+)"
)

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            mean, std, qmin, qmax = map(float, m.groups())
            means.append(mean)
            stds.append(std)
            mins.append(qmin)
            maxs.append(qmax)

means = np.array(means)
stds = np.array(stds)
mins = np.array(mins)
maxs = np.array(maxs)

print("===== Q Statistics =====")
print(f"Number of records: {len(means)}")
print(f"Mean(Q): {means.mean():.4f}")
print(f"Std of batch Mean(Q): {means.std():.4f}")
print(f"Average batch Std(Q): {stds.mean():.4f}")
print(f"Global Min(Q): {mins.min():.4f}")
print(f"Global Max(Q): {maxs.max():.4f}")
print(f"Batch Mean(Q) Min: {means.min():.4f}")
print(f"Batch Mean(Q) Max: {means.max():.4f}")
print(f"Batch Mean(Q) P25: {np.percentile(means, 25):.4f}")
print(f"Batch Mean(Q) P50: {np.percentile(means, 50):.4f}")
print(f"Batch Mean(Q) P75: {np.percentile(means, 75):.4f}")