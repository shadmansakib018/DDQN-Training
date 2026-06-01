import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent
txt_path = script_dir / "ValidationART.txt"

values = []
with open(txt_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                values.append(float(line))
            except ValueError:
                continue

checkpoints = list(range(2000, 2000 + len(values) * 100, 100))

min_val = min(values)
min_idx = values.index(min_val)

colors = ["#e74c3c" if v == min_val else "#3498db" for v in values]

fig, ax = plt.subplots(figsize=(14, 6))

ax.scatter(checkpoints, values, color=colors, s=60, zorder=3)
ax.plot(checkpoints, values, color="#aaaaaa", linewidth=0.8, zorder=2)

ax.scatter(checkpoints[min_idx], min_val, color="#e74c3c", s=120, zorder=4,
           label=f"Lowest: Checkpoint {checkpoints[min_idx]} ({min_val:.3f}s)")
ax.scatter([], [], color="#3498db", s=60, label="Other checkpoints")

ax.set_xlabel("Model Checkpoint", fontsize=13)
ax.set_ylabel("Avg response time (s)", fontsize=13)
ax.set_title("Model Validation: Finding the Optimal Model", fontsize=15, fontweight="bold")
ax.set_xticks(range(2000, checkpoints[-1] + 1, 1000))
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(script_dir / "validation_art.png", dpi=150)
plt.show()
