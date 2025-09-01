import jax.numpy as jnp
import numpy as np

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

    cons_d = jnp.concatenate([jnp.eye(6 * ids["eef_num"]), y], axis = 1)
    cons_h = z

    return cons_d, cons_h

if __name__ == "__main__":
    orien = jnp.ones((EEF_NUM, 3))
    s = jnp.ones([EEF_NUM])

    theta = make_theta(orien, s)
    omega = make_omega(s)

    print(theta)
    print(omega)