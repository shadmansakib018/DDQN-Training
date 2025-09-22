import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib import font_manager

# X-axis: Hours of the day
hours = np.array(range(25))

# Original data
TH = np.array([12.3, 12.3, 12.22, 17.78, 12.39, 12.25, 17.69, 6.71, 43.71, 44.13, 32.76, 17.68, 6.68,
               38.19, 38.05, 12.16, 17.67, 43.78, 32.97, 43.66, 43.59, 43.45, 38.42, 6.78, 12.3])
SBDLB = np.array([8.77, 8.62, 8.77, 13.38, 8.83, 8.74, 13.1, 4.76, 38.51, 38.13, 27.33, 13.32, 4.7,
                  32.26, 32.25, 8.72, 13.18, 38.08, 26.87, 37.71, 37.61, 37.67, 32.36, 4.76, 8.77])
RL_M50 = np.array([5.18, 5.26, 5.09, 9.37, 5.35, 5.42, 8.57, 3.5, 30.65, 30.02, 21.91, 8.76, 3.48,
                   26.49, 25.68, 5.13, 8.78, 30.36, 21.13, 29.89, 29.71, 29.34, 25.32, 3.57, 5.18])
RL_M150 = np.array([6.09, 5.78, 5.84, 9.35, 6.14, 7.3, 9.27, 4.16, 31.42, 30.67, 23.9, 10.7, 4.08,
                    26.94, 27.49, 6.28, 9.39, 30.78, 23.49, 31.21, 31.49, 30.81, 27.56, 4.52, 6.09])

# Create smoother curves using spline interpolation
x_smooth = np.linspace(hours.min(), hours.max(), 300)

def smooth_line(x, y):
    spline = make_interp_spline(x, y, k=3)
    return spline(x_smooth)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x_smooth, smooth_line(hours, TH), label='TH', linewidth=2)
plt.plot(x_smooth, smooth_line(hours, SBDLB), label='SBDLB', linewidth=2)
plt.plot(x_smooth, smooth_line(hours, RL_M50), label='RL-M50', linewidth=2)
plt.plot(x_smooth, smooth_line(hours, RL_M150), label='RL-M150', linewidth=2)

# Labels and title with bold text
plt.xlabel('Hour of Day', fontsize=14, fontweight='bold')
plt.ylabel('Avg Response Time (seconds)', fontsize=14, fontweight='bold')
plt.title('Hourly Average Response Times Using 10VMS', fontsize=16, fontweight='bold')

# Bold legend
plt.legend(fontsize=12, prop={'weight': 'bold'})
plt.grid(axis='y', color='lightgray')

# Bold axis ticks
plt.xticks(hours, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
