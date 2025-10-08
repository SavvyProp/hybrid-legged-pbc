import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt("data/impulse_rew.csv", delimiter=",")
data_pd = np.loadtxt("data/pd_impulse_rew.csv", delimiter=",")

data = np.clip(data, 0.0, None)
data_pd = np.clip(data_pd, 0.0, None)

plt.plot(np.arange(len(data)) * 30, data, label="FF + PD")
plt.plot(np.arange(len(data_pd)) * 30, data_pd, label="PD")
plt.xlabel("Impulse Force (N)")
plt.ylabel("Total Reward")
plt.legend()
plt.title("Impulse Force vs Reward")
plt.show()