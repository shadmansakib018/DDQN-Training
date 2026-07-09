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

plt.figure(figsize=(12, 6))
plt.plot(checkpoints, values, label="Average Response Time", marker='o')

# Highlight the minimum value with a red marker
plt.scatter(checkpoints[min_idx], min_val, color='red', zorder=5, label=f"Lowest value: {min_val:.3f} s")

# Title and labels with bold font weight
plt.title("Model Validation: Finding the Optimal Response Time", fontsize=16)
plt.xlabel("Model Checkpoint", fontsize=16)
plt.ylabel("Average Response Time (s)", fontsize=16)

# Add a legend
plt.legend()

# Make axis tick labels bold
plt.tick_params(axis='both', labelsize=14)

plt.savefig(script_dir / "validation_art.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.grid(True, color='lightgray')
plt.show()
