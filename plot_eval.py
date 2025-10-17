import numpy as np
import matplotlib.pyplot as plt
def plot_impulse():
    ft = np.genfromtxt("data/eval/ftk3_impulse_frc_1dt_success_rate.csv", delimiter=",")
    pd = np.genfromtxt("data/eval/pd32_impulse_frc_1dt_success_rate.csv", delimiter=",")
    plt.plot(ft[:, 0], ft[:, 1], label="PD + Centroidal FF")
    plt.plot(pd[:, 0], pd[:, 1], label="PD")
    plt.xlabel("Impulse Force (N)")
    plt.ylabel("Success Rate")
    plt.title("Impulse Force Disturbance Evaluation For 32 Envs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_impulse()