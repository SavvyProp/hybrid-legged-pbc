import os
import numpy as np
import matplotlib.pyplot as plt


def load_matrix(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.loadtxt(path, delimiter=",")



def main():
    pd_tau_path = os.path.join('data', 'pd_tau.csv')
    u_path = os.path.join('data', 'u.csv')

    pd_tau = load_matrix(pd_tau_path)
    u = load_matrix(u_path)
    f = load_matrix(os.path.join('data', 'f.csv'))
    q_ddot_com = load_matrix(os.path.join('data', 'q_ddot_com.csv'))
    com_ref = load_matrix(os.path.join('data', 'com_ref.csv'))
    real_com_vel = load_matrix(os.path.join('data', 'real_com_vel.csv'))
    des_com_vel = load_matrix(os.path.join('data', 'des_com_vel.csv'))
    des_com_angvel = load_matrix(os.path.join('data', 'des_angvel.csv'))
    real_angvel = load_matrix(os.path.join('data', 'real_angvel.csv'))

    range_lower = 100
    range_upper = 200
    
    #plt.plot(pd_tau[range_lower:range_upper, 11])
    #plt.plot(u[range_lower:range_upper, 11])
    lf_mag = np.linalg.norm(f[:, 0:3], axis=1)
    lt_mag = np.linalg.norm(f[:, 3:6], axis=1)
    rf_mag = np.linalg.norm(f[:, 6:9], axis=1)
    rt_mag = np.linalg.norm(f[:, 9:12], axis=1)
    #plt.plot(lt_mag)
    #plt.plot(rt_mag)

    #des_com_vel_mag = np.linalg.norm(des_com_vel, axis=1)
    #plt.plot(des_com_vel_mag)
    #plt.plot(q_ddot_com[range_lower:range_upper, 0], label='x')
    #plt.plot(q_ddot_com[range_lower:range_upper, 1], label='y')
    #plt.plot(q_ddot_com[:, 2], label='z')
    #plt.plot(com_ref[range_lower:range_upper, 0], label='ref x')
    #plt.plot(com_ref[range_lower:range_upper, 1], label='ref y')
    #plt.plot(com_ref[range_lower:range_upper, 2], label='ref z')
    plt.plot(real_com_vel[range_lower:range_upper, 0], label='real z')
    plt.plot(des_com_vel[range_lower:range_upper, 0], label='des z')
    #plt.plot(des_com_angvel[range_lower:range_upper, 0], label='des yaw')
    #plt.plot(real_angvel[range_lower:range_upper, 0], label='real yaw')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
