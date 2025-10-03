import jax.numpy as jnp

def schur_solve(qp_q, qp_c, cons_lhs, cons_rhs):
    Q = 0.5 * (qp_q + qp_q.T)
    A = cons_lhs
    c = qp_c
    b = cons_rhs
    Z = jnp.zeros((A.shape[0], A.shape[0]), dtype=jnp.float32)
    KKT = jnp.block([[Q, A.T],
                     [A, Z]])
    rhs = jnp.concatenate([c, b], axis=0)
    sol_all = jnp.linalg.solve(KKT, rhs)
    sol = sol_all[:Q.shape[0]]
    return sol