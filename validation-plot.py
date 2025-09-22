import matplotlib.pyplot as plt
import numpy as np
 
# Given y-axis data
y_values = [
    9.371, 5.867, 5.289, 5.084, 5.592, 4.8, 5.022, 4.192, 3.945, 5.252, 4.933, 4.68, 4.528,
    5.486, 4.99, 5.733, 5.442, 4.669, 4.33, 4.498, 4.59, 3.42, 4.507, 5.213, 3.524, 4.591,
    4.776, 3.837, 5.624, 3.829, 4.126, 4.279, 3.33, 5.481, 5.016, 5.371, 3.814, 3.675, 3.452,
    3.842, 4.137, 3.539, 3.575, 3.818, 3.751, 4.16, 4.747, 3.882, 4.563, 4.135, 4.351, 4.243,
    3.706, 3.827, 5.441, 3.992, 4.097, 4.579, 3.797, 3.385, 3.446
]
 
# X values for "checkpoint" ranging from 2000 to 8000 in 61 steps
x_values = np.linspace(2000, 8000, 61)
 
# Find the index of the minimum y-value for highlighting
min_y = min(y_values)
min_y_index = y_values.index(min_y)
 
# Plotting the graph
# Plotting the graph
plt.figure(figsize=(12,6))
plt.plot(x_values, y_values, label="Average Response Time", marker='o')
 
# Highlight the minimum value with a red marker
plt.scatter(x_values[min_y_index], min_y, color='red', zorder=5, label=f"Lowest value: {min_y:.3f} s")
 
# Title and labels with bold font weight
plt.title("Model Validation: Finding the Optimal Response Time", fontsize=16)
plt.xlabel("Model Checkpoint", fontsize=16)
plt.ylabel("Average Response Time (s)", fontsize=16)
 
# Add a legend
plt.legend()
 
# Make axis tick labels bold
plt.tick_params(axis='both', labelsize=14)
 
plt.savefig("model_validation_plot.png", dpi=300, bbox_inches="tight")
 
 
# Show the plot
plt.grid(True, color='lightgray')
plt.show()