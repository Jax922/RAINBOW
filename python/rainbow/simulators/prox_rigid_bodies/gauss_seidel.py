import numpy as np
import numba as nb
from rainbow.util.timer import Timer

@nb.njit
def prox_sphere_nb(z_s, z_t, z_tau, mu, x_n):
    """
    Proximal point of z to a sphere.

    :param z_s:       s-component of current z-point.
    :param z_t:       t-component of current z-point.
    :param z_tau:     tau-component of current z-point.
    :param mu:        The coefficient of friction.
    :param x_n:       The current normal force magnitude.
    :return:          The new z-point which will be the closest point to the sphere with radius my*x_n.
    """
    if x_n <= 0.0:
        return 0.0, 0.0, 0.0
    radius = mu * x_n
    sqr_z_norm = z_s * z_s + z_t * z_t + z_tau * z_tau
    if sqr_z_norm <= radius * radius:
        return z_s, z_t, z_tau
    scale = radius / np.sqrt(sqr_z_norm)
    return z_s * scale, z_t * scale, z_tau * scale

def prox_sphere(z_s, z_t, z_tau, mu, x_n):
    """
    Proximal point of z to a sphere.

    :param z_s:       s-component of current z-point.
    :param z_t:       t-component of current z-point.
    :param z_tau:     tau-component of current z-point.
    :param mu:        The coefficient of friction.
    :param x_n:       The current normal force magnitude.
    :return:          The new z-point which will be the closest point to the sphere with radius my*x_n.
    """
    if x_n <= 0.0:
        return 0.0, 0.0, 0.0
    radius = mu * x_n
    sqr_z_norm = z_s * z_s + z_t * z_t + z_tau * z_tau
    if sqr_z_norm <= radius * radius:
        return z_s, z_t, z_tau
    scale = radius / np.sqrt(sqr_z_norm)
    return z_s * scale, z_t * scale, z_tau * scale


def prox_origin(z_s, z_t, z_tau, mu, x_n):
    """
    Proximal point of z to the origin.

    :param z_s:       s-component of current z-point.
    :param z_t:       t-component of current z-point.
    :param z_tau:     tau-component of current z-point.
    :param mu:        The coefficient of friction.
    :param x_n:       The current normal force magnitude.
    :return:          The new z-point which will be the closest point to the sphere with radius my*x_n.
    """
    return 0.0, 0.0, 0.0


def prox_ellipsoid(engine, z_s, z_t, z_tau, mu_s, mu_t, mu_tau, x_n):
    # 2022-05-09 Kenny TODO: This interface is currently not compatible with
    #                   the friction_solver interface that we assume in the
    #                   sweep function.
    """

    :param engine:
    :param z_s:
    :param z_t:
    :param z_tau:
    :param mu_s:
    :param mu_t:
    :param mu_tau:
    :param x_n:
    :return:
    """
    if x_n <= 0.0:
        return 0.0, 0.0, 0.0

    # If surface friction is frictionless, just return prox origin
    if mu_s == 0 and mu_t == 0 and mu_tau == 0:
        return prox_origin(z_s, z_t, z_tau, mu_s, x_n)

    if mu_s == mu_t and mu_s == mu_tau:
        return prox_sphere(z_s, z_t, z_tau, mu_s, x_n)

    a = mu_s * x_n
    b = mu_t * x_n
    c = mu_tau * x_n

    scale = 1 / max(1, a, b, c, abs(z_s), abs(z_t), abs(z_tau))
    sa, sb, sc, sx, sy, sz = (
        scale * a,
        scale * b,
        scale * c,
        scale * z_s,
        scale * z_t,
        scale * z_tau,
    )

    # Precompute squared values
    saa, sbb, scc, sxx, syy, szz = sa * sa, sb * sb, sc * sc, sx * sx, sy * sy, sz * sz

    # Check if point is already on or inside ellipsoid
    f0 = (sxx / saa) + (syy / sbb) + (szz / scc) - 1
    if f0 < engine.params.ellipsoid_tolerance:
        return z_s, z_t, z_tau

    t0 = 0
    t1 = max(sa, sb, sc) * np.linalg.norm([sx, sy, sz])
    g0 = (
        (saa * sxx) / ((saa + t0) ** 2)
        + (sbb * syy) / ((sbb + t0) ** 2)
        + (scc * szz) / ((scc + t0) ** 2)
        - 1
    )
    g1 = (
        (saa * sxx) / ((saa + t1) ** 2)
        + (sbb * syy) / ((sbb + t1) ** 2)
        + (scc * szz) / ((scc + t1) ** 2)
        - 1
    )
    while g1 > 0:
        t1 *= engine.params.ellipsoid_expansion
        g1 = (
            (saa * sxx) / ((saa + t1) ** 2)
            + (sbb * syy) / ((sbb + t1) ** 2)
            + (scc * szz) / ((scc + t1) ** 2)
            - 1
        )

    tk = (t0 + t1) * 0.5
    for iteration in range(engine.params.ellipsoid_max_iterations):
        # Stagnation test
        if abs(t1 - t0) < engine.params.ellipsoid_tolerance:
            break
        gk = (
            (saa * sxx) / ((saa + tk) ** 2)
            + (sbb * syy) / ((sbb + tk) ** 2)
            + (scc * szz) / ((scc + tk) ** 2)
            - 1
        )
        # Absolute convergence test
        if abs(gk) < engine.params.ellipsoid_tolerance:
            break
        if gk > 0:
            t0 = tk
        else:
            t1 = tk
        tk = (t0 + t1) * 0.5

    tk /= scale * scale

    x_s = (a * a * z_s) / (a * a + tk)
    x_t = (b * b * z_t) / (b * b + tk)
    x_tau = (c * c * z_tau) / (c * c + tk)

    return x_s, x_t, x_tau

# """
    # Compute the matrix-vector product M*v for a CSR matrix M.

    # :param M_data:       The data array of the CSR matrix M.
    # :param M_indices:    The indices array of the CSR matrix M.
    # :param M_inputr:     The inputr array of the CSR matrix M.
    # :param v:            The vector v.
    # :return:             The matrix-vector product M*v.
    # """

@nb.njit
def csr_matvec(M_data, M_indices, M_indptr, v):
    """ Compute the matrix-vector product M*v for a CSR matrix M.

    Args:
        M_data (List): The data array of the CSR matrix M.
        M_indices (List): The indices array of the CSR matrix M.
        M_inputr (List): The inputr array of the CSR matrix M.
        v (List): The vector v.

    Returns:
        List: The matrix-vector product M*v
    """
    # result = np.zeros(M_indptr.shape[0] - 1, dtype=v.dtype)
    # # parallel computing the matrix-vector product
    # for i in nb.prange(result.shape[0]):
    #     result[i] = np.sum(M_data[M_indptr[i]:M_indptr[i + 1]] * v[M_indices[M_indptr[i]:M_indptr[i + 1]]])
    # return result
    rows = M_indptr.shape[0] - 1
    result = np.zeros(rows, dtype=v.dtype)
    
    for i in nb.prange(rows):
        start_idx = M_indptr[i]
        end_idx = M_indptr[i + 1]
        result[i] = np.dot(M_data[start_idx:end_idx], v[M_indices[start_idx:end_idx]])
    return result


@nb.njit
def _get_indices_of_row_nb(M_indptr):
    starts = M_indptr[:-1]
    ends = M_indptr[1:]
    return starts, ends


@nb.njit
def extract_columns_from_csr_nb(data, indices, ind_ptr, cols):
    """
    Extract specified columns from the data underlying a CSR matrix using Numba.

    Args:
        data (array): Data array of the CSR matrix.
        indices (array): Indices array of the CSR matrix.
        indptr (array): Indptr array of the CSR matrix.
        cols (list or array): The columns to extract.

    Returns:
        tuple: New data, indices, and indptr arrays for the extracted columns.
    """
    new_data = []
    new_indices = []
    new_indptr = [0]
    
    for i in range(len(ind_ptr)-1):
        start_idx = ind_ptr[i]
        end_idx = ind_ptr[i+1]
        for j in range(start_idx, end_idx):
            if indices[j] in cols:
                new_data.append(data[j])
                new_indices.append(indices[j]-cols[0])
    
        new_indptr.append(len(new_data))

    return np.array(new_data), np.array(new_indices), np.array(new_indptr)


# @nb.njit(parallel=True)
# def _extract_block(x, r, b, mu, K):
#     N = 4
#     x_b = np.empty((K, N), dtype=x.dtype)
#     r_b = np.empty((K, N), dtype=r.dtype)
#     b_b = np.empty((K, N), dtype=b.dtype)
#     mu_k = np.empty(K, dtype=mu.dtype)
#     for k in nb.prange(K):
#         x_b[k, :] = x[N*k:N*(k + 1)]
#         r_b[k, :] = r[N*k:N*(k + 1)]
#         b_b[k, :] = b[N*k:N*(k + 1)]
#         mu_k[k] = mu[k]
#     return x_b, r_b, b_b, mu_k


@nb.njit
def sweep_nb(K, J_data, J_indices, J_indptr, WJT_data, WJT_indices, WJT_indptr, b, mu, r, x, friction_solver):
    w = csr_matvec(WJT_data, WJT_indices, WJT_indptr, x)
    for k in range(K):
        block = np.arange(4 * k, 4 * k + 4)
        mu_k = mu[k]  # Only isotropic Coulomb friction
        x_b = x[block]
        delta = (
            x_b.copy()
        )  # Used to keep the old values and compute the change in values
        r_b = r[block]
        b_b = b[block]
        # By definition
        #       z = x - r (J WJ^T x  + b)
        #         = x - r ( A x  + b)
        # We use
        #        w =  WJ^T x
        # so
        #       z  = x - r ( J w  + b)
        z_b = x_b - np.multiply(r_b, (csr_matvec(J_data, J_indices, J_indptr, w)[block] + b_b))

        # Solve: x_n = prox_{R^+}( x_n - r (A x_n + b) )
        x_b[0] =max(0.0, z_b[0])

        # Solve: x_f = prox_C( x_f - r (A x_f + b))
        x_b[1], x_b[2], x_b[3] = friction_solver(z_b[1], z_b[2], z_b[3], mu_k, x_b[0])
        # Put updated contact forces back into solution vector
        x[block] = x_b
        # Get the change in the x_block
        np.subtract(x_b, delta, delta)
        # Updating w so it reflect the change in x, remember w = WJT delta
        WJT_b = extract_columns_from_csr_nb(WJT_data, WJT_indices, WJT_indptr, block)
        w += csr_matvec(WJT_b[0], WJT_b[1], WJT_b[2], delta)

    return x
    

def sweep(K, J, WJT, b, mu, r, x, friction_solver, engine):
    """

    :param K:
    :param J:
    :param WJT:
    :param b:
    :param mu:
    :param r:
    :param x:
    :param friction_solver:
    :param engine:
    :return:
    """
    w = WJT.dot(x)
    # w_test = csr_matvec(WJT.data, WJT.indices, WJT.indptr, x)
    # close_test = np.allclose(w, w_test, atol=1e-8)
    # if not close_test:
    #     print("unequal")

    for k in range(K):
        block = range(4 * k, 4 * k + 4)
        mu_k = mu[k]  # Only isotropic Coulomb friction
        x_b = x[block]
        delta = (
            x_b.copy()
        )  # Used to keep the old values and compute the change in values
        r_b = r[block]
        b_b = b[block]
        # By definition
        #       z = x - r (J WJ^T x  + b)
        #         = x - r ( A x  + b)
        # We use
        #        w =  WJ^T x
        # so
        #       z  = x - r ( J w  + b)
        z_b = x_b - np.multiply(r_b, (J.dot(w)[block] + b_b))

        # Solve:         x_n = prox_{R^+}( x_n - r (A x_n + b) )
        x_b[0] = np.max([0.0, z_b[0]])

        # Solve:         x_f = prox_C( x_f - r (A x_f + b))
        x_b[1], x_b[2], x_b[3] = friction_solver(z_b[1], z_b[2], z_b[3], mu_k, x_b[0])
        # Put updated contact forces back into solution vector
        x[block] = x_b
        # Get the change in the x_block
        np.subtract(x_b, delta, delta)
        # Updating w so it reflect the change in x, remember w = WJT delta
        # TODO 2020-08-17 Kristian: WJT is in bsr matrix format, which does not support indexing and we can therefore
        #  not access the block sub-matrix. Currently we circumvent this by converting it to a csr matrix instead,
        #  however another solution might be better.
        # print(type(WJT))
        # t1 = WJT[:, np.array(block, dtype=np.int32)]
        # t2 = extract_columns_from_csr_nb(WJT.data, WJT.indices, WJT.indptr, np.array(block, dtype=np.int32))
        # WJT_b_data = t2[0]
        # WJT_b_indices = t2[1]
        # if(t1.data.shape != WJT_b_data.shape):
        #     print("data shape diff")
        #     print(t1.data.shape, WJT_b_data.shape)
        # else:
        #     np.testing.assert_allclose(t1.data, WJT_b_data, atol=1e-8)
        #     close_test_1 = np.allclose(t1.data, WJT_b_data, atol=1e-8)
        #     if not close_test_1:
        #         print("data unequal")

        # if(t1.indices.shape != WJT_b_indices.shape):
        #     print("indices shape diff")
        #     print(t1.indices.shape, WJT_b_indices.shape)
        # else:
        #     np.testing.assert_allclose(t1.indices, WJT_b_indices, atol=1e-8)
        #     close_test_2 = np.allclose(t1.indices, WJT_b_indices, atol=1e-8)
        #     if not close_test_2:
        #         print("indices unequal")
        
        # if(t1.indptr.shape != t2[2].shape):
        #     print("indptr shape diff")
        #     print(t1.indptr.shape, t2[2].shape)
        # else:
        #     np.testing.assert_allclose(t1.indptr, t2[2], atol=1e-8)
        #     close_test_3 = np.allclose(t1.indptr, t2[2], atol=1e-8)
        #     if not close_test_3:
        #         print("indptr unequal")
        #         print("t1->", t1.indptr)
        #         print("t2->", t2[2])
        w += WJT[:, block].dot(delta)
        # w += t1.dot(delta)
    return x


def solve(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix):
    """

    :param J:
    :param WJT:
    :param b:
    :param mu:
    :param friction_solver:
    :param engine:
    :param stats:
    :param debug_on:
    :param prefix:
    :return:
    """
    timer = None
    if debug_on:
        timer = Timer("Gauss Seidel")
        stats[prefix + "residuals"] = (
            np.ones(engine.params.max_iterations, dtype=np.float64) * np.inf
        )
        stats[prefix + "lambda"] = np.zeros(
            [engine.params.max_iterations] + list(b.shape), dtype=np.float64
        )
        stats[prefix + "reject"] = np.zeros(engine.params.max_iterations, dtype=bool)
        stats[prefix + "exitcode"] = 0
        stats[prefix + "iterations"] = engine.params.max_iterations
        timer.start()

    K = len(engine.contact_points)

    x = np.zeros(b.shape, dtype=np.float64)  # The current iterate
    sol = np.zeros(
        b.shape, dtype=np.float64
    )  # The last best known solution, used for restarting if divergence
    error = np.zeros(b.shape, dtype=np.float64)  # The residual vector

    # Compute initial r-factor value
    delassus_diag = np.sum(J.multiply(WJT.T), axis=1).A1
    delassus_diag[delassus_diag == 0] = 1
    r = 1.0 / delassus_diag

    # Extract parameter values for controlling the adaptive r-factor strategy
    nu_reduce = engine.params.nu_reduce
    nu_increase = engine.params.nu_increase
    too_small_merit_change = engine.params.too_small_merit_change

    last_merit = np.Inf

    for iteration in range(engine.params.max_iterations):
        if engine.params.proximal_speedup:
            x = sweep_nb(K, J.data, J.indices, J.indptr, WJT.data, WJT.indices, WJT.indptr, b, mu, r, x, friction_solver)
        else:
            x = sweep(K, J, WJT, b, mu, r, x, friction_solver, engine)

        np.subtract(x, sol, error)
        merit = np.linalg.norm(error, np.inf)
        if debug_on:
            stats[prefix + "lambda"][iteration] = x
            stats[prefix + "residuals"][iteration] = merit
        # Test stopping criteria
        if merit < engine.params.absolute_tolerance:
            if debug_on:
                stats[prefix + "iterations"] = iteration
                stats[prefix + "exitcode"] = 1
                timer.end()
                stats[prefix + "solver_time"] = timer.elapsed
            return x, stats
        if np.abs(merit - last_merit) < engine.params.relative_tolerance * last_merit:
            if debug_on:
                stats[prefix + "iterations"] = iteration
                stats[prefix + "exitcode"] = 2
                timer.end()
                stats[prefix + "solver_time"] = timer.elapsed
            return x, stats

        # Update r-factors
        if merit > last_merit:
            # Divergence detected: reduce R-factor and roll-back solution to last known good iterate!
            np.multiply(nu_reduce, r, r)
            np.copyto(x, sol)
            if debug_on:
                stats[prefix + "reject"][iteration] = True
        else:
            if last_merit - merit < too_small_merit_change:
                # Convergence is slow: increase r-factor
                np.multiply(nu_increase, r, r)
            # Convergence detected: accept x as better solution
            last_merit = merit
            np.copyto(sol, x)

    # If this point of the code is reached then it means the method did not converge within the given iterations.
    if debug_on:
        timer.end()
        stats[prefix + "solver_time"] = timer.elapsed
    return sol, stats
