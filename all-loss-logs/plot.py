import numpy as np
import matplotlib.pyplot as plt

# Files and labels
files = {
    "AC": "AC.txt",
    "A2C": "A2C.txt",
    "A3C": "A3C.txt",
    "DDQN": "DDQN.txt",
    "PPO": "PPO.txt"
}


def load_data(filename):
    """
    Reads a text file where each line contains:
    loss,reward
    """
    data = np.loadtxt(filename, delimiter=",")
    loss = data[:, 0]
    reward = data[:, 1]
    return loss, reward


def exponential_moving_average(data, alpha=0.01):
    ema = [data[0]]

    for i in range(1, len(data)):
        new_value = alpha * data[i] + (1 - alpha) * ema[-1]
        ema.append(new_value)

    return np.array(ema)


# --------------------------
# Running Reward Plot
# --------------------------
plt.figure(figsize=(12, 6))

for name, file in files.items():
    loss, reward = load_data(file)
    running_reward = running_reward = exponential_moving_average(reward, alpha=0.01)

    plt.plot(running_reward, linewidth=2, label=name)

plt.xlabel("Epoch", fontsize=14, fontweight="bold")
plt.ylabel("Running Reward", fontsize=14, fontweight="bold")

plt.legend(prop={"weight": "bold", "size": 12})

plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("Running_Reward_Comparison.png", dpi=300)
plt.show()


# --------------------------
# Running Loss Plot
# --------------------------
plt.figure(figsize=(12, 6))

for name, file in files.items():
    loss, reward = load_data(file)
    running_loss = running_loss = exponential_moving_average(loss, alpha=0.01)

    plt.plot(running_loss, linewidth=2, label=name)

plt.xlabel("Epoch", fontsize=14, fontweight="bold")
plt.ylabel("Running Loss", fontsize=14, fontweight="bold")

plt.legend(prop={"weight": "bold", "size": 12})

plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("Running_Loss_Comparison.png", dpi=300)
plt.show()