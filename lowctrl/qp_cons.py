import jax.numpy as jnp
import numpy as np
import flax.linen as nn

EEF_NUM = 4

def vec2diags(v, ids):
    # v is a vector of length N
    # returns a matrix of shape (6N, 6N) with the elements of v repeated on the diagonals
    n = ids["eef_num"]
    D = jnp.zeros((n * 6, n * 6))
    rows = jnp.arange(n) * 6
    cols = rows
    D = D.at[rows[:, None] + jnp.arange(6)[None, :], cols[:, None] + jnp.arange(6)[None, :]].set(v[:, None])
    return D

def make_theta(oriens, s, ids):
    # oriens is an N by 4 matrix
    oriens = oriens * s[:, None]
    n = ids["eef_num"]
    theta = jnp.zeros((n * 6, n * 6))
    blocks = oriens[:, :, None] @ oriens[:, None, :]
    starts = jnp.arange(n) * 6
    rows = starts[:, None] + jnp.arange(3)[None, :]   # (n, 3)
    cols = rows
    theta = theta.at[rows[:, :, None], cols[:, None, :]].set(blocks)
    return theta

def make_omega(w, ids):
    weights = jnp.exp(-1 * w)
    omega = vec2diags(weights, ids)
    return omega

def make_omega2(w, ids):
    weights = jnp.exp(w)
    omega = vec2diags(weights, ids)
    return omega


def make_acc_cons(w, ju, lstsq_opt, ids):
    omega = make_omega2(w, ids)
    ju = omega @ ju
    lstsq_opt = omega @ lstsq_opt
    big_q_a = 2 * ju.T @ ju
    big_q_a += jnp.eye(6) * 1e-6
    small_q_a = 2 * ju.T @ lstsq_opt
    return big_q_a, small_q_a

def qp_solve(m, h, 
            qp_weights, oriens, w, 
            jacs, jvp, 
            a_stc, qc,
            ids,
            ):
    s = nn.sigmoid(w)
    theta = make_theta(oriens, s, ids)
    omega = make_omega(w, ids)
    q_frc = theta * qp_weights[0] + omega * qp_weights[1]

    ju = jacs[:, :6]
    jc = jacs[:, 6:]
    lstsq_opt = a_stc - jvp - jc @ qc

    big_q_a, small_q_a = make_acc_cons(w, ju, lstsq_opt, ids)
    #r_a = lstsq_opt[None, :] @ lstsq_opt[:, None]

    # Stack is q on top followed by f

    f_size = ids["eef_num"] * 6

    big_q = jnp.block([[big_q_a, jnp.zeros([6, f_size])],
                       [jnp.zeros([f_size, 6]), q_frc]])
    small_q = jnp.concatenate([small_q_a, jnp.zeros([f_size])], axis = 0)

    # Setup constraints
    i_q = jnp.concatenate([jnp.eye(6), jnp.zeros([6, f_size])], axis = 1)
    i_frc = jnp.concatenate([jnp.zeros([f_size, 6]), jnp.eye(f_size)], axis = 1)

    #Cons: -juT @ F + m_uu @ q_u + muc @ q_c + h_u = 0
    # juT @ I_f @ F - m_uu @ I_q @ q_u = h_u + muc @ q_c
    h_u = h[:6]
    m_uu = m[:6, :6]
    m_uc = m[:6, 6:]
    cons_a = ju.T @ i_frc - m_uu @ i_q
    cons_b = h_u + m_uc @ qc

    # Solve the QP problem

    v1 = jnp.linalg.solve(big_q, small_q)
    v2 = jnp.linalg.solve(big_q, cons_a.T)
    v3 = cons_a @ v2
    v4 = cons_a @ v1 - cons_b

    sol = v1 - v2 @ jnp.linalg.solve(v3, v4)

    q_u = sol[:6]
    f = sol[6:]
    return f, q_u



def qp_cons(m_u_uc, h_u, qp_weights,
            oriens, s, w, ju, ids):
    theta = make_theta(oriens, s, ids)
    omega = make_omega(w, ids)

    q = theta * qp_weights[0] + omega * qp_weights[1]
    qinv = jnp.linalg.inv(q)

    sol1 = jnp.matmul(ju.T, jnp.matmul(qinv, ju))
    sol2 = jnp.matmul(qinv, 
                      jnp.matmul(ju, 
                                 jnp.linalg.inv(sol1)))
    y = sol2 @ m_u_uc
    z = sol2 @ h_u

    #cons_d = jnp.concatenate([jnp.eye(6 * ids["eef_num"]), y], axis = 1)
    cons_d = y
    cons_h = z

    return cons_d, cons_h

if __name__ == "__main__":
    orien = jnp.ones((EEF_NUM, 3))
    s = jnp.ones([EEF_NUM])

    theta = make_theta(orien, s)
    omega = make_omega(s)

    print(theta)
    print(omega)