import matplotlib.pyplot as plt
import numpy as np

base_tau = np.loadtxt("data/base_tau.csv", delimiter=",")
inv_tau = np.loadtxt("data/inv_tau.csv", delimiter=",")
pinv_tau = np.loadtxt("data/pinv_tau.csv", delimiter=",")

ind = 14
a = 200
b = 300
joint_base = base_tau[a:b, ind]
joint_inv = inv_tau[a:b, ind]
joint_pinv = pinv_tau[a:b, ind]

plt.plot(joint_base, label = "base")
plt.plot(joint_inv, label = "inv")
plt.plot(joint_pinv, label = "pinv")
plt.legend()
plt.show()