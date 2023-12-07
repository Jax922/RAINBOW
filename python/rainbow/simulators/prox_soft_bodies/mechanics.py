import rainbow.math.matrix3 as M3
import numpy as np

def svd_rotation_variant(F):
    """ 
    Perform a rotation variant of the Singular Value Decomposition (SVD) on a matrix.

    This function computes the SVD of a 3x3 matrix, with additional steps to handle rotations
    and ensure that the U and V matrices do not contain reflections. If a reflection is detected
    in U or V, it is corrected by adjusting the corresponding singular values.

    :param F: deformation gradient.
    :return: A tuple containing three elements:
        U (numpy.ndarray): A 3x3 orthogonal matrix U from the SVD.
        Sigma (numpy.ndarray): A 3x3 diagonal matrix of singular values.
        V (numpy.ndarray): A 3x3 orthogonal matrix V from the SVD.
    """
    U, Sigma, Vt = np.linalg.svd(F)
    U, V = U, Vt.T

    L = np.identity(3)
    L[2, 2] = np.linalg.det(U @ V.T)

    detU = np.linalg.det(U)
    detV = np.linalg.det(V)

    if detU < 0 and detV > 0:
        U = U @ L
    if detU > 0 and detV < 0:
        V = V @ L

    Sigma[2] *= L[2, 2]
    return U, Sigma, V

def build_twist_and_flip_eigenvectors(U, V):
    sqrt_2_inv = 1.0 / np.sqrt(2.0)

    # Twist matrices
    T0 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])  # x-twist
    T1 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])  # y-twist
    T2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])  # z-twist

    Q0 = sqrt_2_inv * (U @ T0 @ V.T)
    Q1 = sqrt_2_inv * (U @ T1 @ V.T)
    Q2 = sqrt_2_inv * (U @ T2 @ V.T)

    # Flip matrices
    L0 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])  # x-flip
    L1 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])  # y-flip
    L2 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # z-flip

    Q3 = sqrt_2_inv * (U @ L0 @ V.T)
    Q4 = sqrt_2_inv * (U @ L1 @ V.T)
    Q5 = sqrt_2_inv * (U @ L2 @ V.T)

    # Flatten matrices and combine
    Q = np.column_stack([Q0.flatten(), Q1.flatten(), Q2.flatten(),
                         Q3.flatten(), Q4.flatten(), Q5.flatten()])
    return Q


def build_scaling_eigenvectors(U, Q, V):
    # 提取 Q 矩阵的列向量
    q0, q1, q2 = Q[:, 0], Q[:, 1], Q[:, 2]

    # 计算缩放模式矩阵
    Q0 = U @ np.diag(q0) @ V.T
    Q1 = U @ np.diag(q1) @ V.T
    Q2 = U @ np.diag(q2) @ V.T

    # 将矩阵展平为一维数组，并合并到 Q9 中
    Q9 = np.column_stack([Q0.flatten(), Q1.flatten(), Q2.flatten()])
    return Q9



def right_cauchy_strain_tensor(F):
    """
    This function computes the right Cauchy strain tensor.

    :param F:  The deformation gradient
    :return:   The right Cauchy strain tensor
    """
    return np.matmul(np.transpose(F), F)


def green_strain_tensor(F):
    """
    This function computes the right Green strain tensor.

    :param F:  The deformation gradient
    :return:   The Green strain tensor
    """
    C = right_cauchy_strain_tensor(F)
    E = (C - M3.identity()) / 2
    return E


def cauchy_stress_tensor(F, S):
    """
    This function computes the right Cauchy stress tensor.

    :param F:   The deformation gradient.
    :param S:   The 2nd Piola-Kirchoff stress tensor.
    :return:    The Cauchy stress tensor.
    """
    j = np.linalg.det(F)
    sigma = np.multiply(
        (1 / j)[:, None, None], np.matmul(F, (np.matmul(S, F.transpose(0, 2, 1))))
    )
    return sigma


def create_material_parameters(name=None):
    """
    Convenience function to quickly get some "sensible" material parameters.

    :param name:  The name of the material to create parameters for.
    :return:      A triplet of Young modulus, Poisson ratio, and mass density.
    """
    E = 10e5  # Young modulus
    nu = 0.3  # Poisson ratio
    rho = 1000  # Mass density
    if name is None:
        return E, nu, rho
    if name.lower() == "cartilage":
        E = 0.69e6
        nu = 0.018
        rho = 1000
    if name.lower() == "cortical bone":
        E = 16.16e9
        nu = 0.33
        rho = 1600
    if name.lower() == "cancellous bone":
        E = 452e6
        nu = 0.3
        rho = 1600
    if name.lower() == "rubber":
        E = 0.01e9
        nu = 0.48
        rho = 1050
    if name.lower() == "concrete":
        E = 30e9
        nu = 0.20
        rho = 2320
    if name.lower() == "copper":
        E = 125e9
        nu = 0.35
        rho = 8900
    if name.lower() == "steel":
        E = 210e9
        nu = 0.31
        rho = 7800
    if name.lower() == "aluminium":
        E = 72e9
        nu = 0.34
        rho = 2700
    if name.lower() == "glass":
        E = 50e9
        nu = 0.18
        rho = 2190
    return E, nu, rho


def first_lame(E, nu):
    """
    Convert elastic parameters into Lamé parameters.

    :param E:   Young modulus
    :param nu:  Poisson ratio
    :return:    The corresponding value of the first Lamé parameter (lambda)
    """
    return (nu * E) / ((1 + nu) * (1 - 2 * nu))


def second_lame(E, nu):
    """
    Convert elastic parameters into Lamé parameters.

    :param E:   Young modulus
    :param nu:  Poisson ratio
    :return:    The corresponding value of the second Lamé parameter (mu)
    """
    return E / (2 * (1 + nu))


class SNH:
    """
    Elasticity model from "Stable Neo-Hookean Flesh Simulation".

    https://graphics.pixar.com/library/StableElasticity/paper.pdf
    """

    @staticmethod
    def pk1_stress(F, lambda_in, mu_in):
        """
        Compute stress tensors.

        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The first Piola-Kirchhoff stress tensor
        """
        mu_ = (4.0 / 3.0) * mu_in
        lambda_ = lambda_in + (5.0 / 6.0) * mu_in
        alpha = 1 + (mu_ / lambda_) - (mu_ / (4 * lambda_))
        J = np.linalg.det(F)
        C = np.matmul(np.transpose(F), F)
        I_C = np.trace(C)
        # J = F0 dot (F1 cross F2), from this we can compute dJdF
        dJdF = np.zeros_like(F)
        dJdF[:, 0] = np.cross(F[:, 1], F[:, 2])
        dJdF[:, 1] = np.cross(F[:, 2], F[:, 0])
        dJdF[:, 2] = np.cross(F[:, 0], F[:, 1])
        return mu_ * (1 - (1 / (I_C + 1))) * F + lambda_ * (J - alpha) * dJdF

    @staticmethod
    def energy_density(F, lambda_in, mu_in):
        """
        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The energy density.
        """
        mu_ = (4.0 / 3.0) * mu_in
        lambda_ = lambda_in + (5.0 / 6.0) * mu_in
        alpha = 1 + (mu_ / lambda_) - (mu_ / (4 * lambda_))
        J = np.linalg.det(F)
        C = np.matmul(np.transpose(F), F)
        I_C = np.trace(C)
        return (
            0.5 * mu_ * (I_C - 3)
            + 0.5 * lambda_ * np.square(J - alpha)
            - 0.5 * mu_ * np.log2(I_C + 1)
        )


class SVK:
    """
    Saint Venant Kirchhoff Elastic Model.
    """

    @staticmethod
    def pk1_stress(F, lambda_in, mu_in):
        """
        Compute stress tensors.

        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The first Piola-Kirchhoff stress tensor
        """
        C = np.matmul(F.T, F)
        E = (C - np.eye(3)) / 2
        S = lambda_in * np.trace(E) * np.eye(3) + 2 * mu_in * E
        return F.dot(S)

    @staticmethod
    def energy_density(F, lambda_in, mu_in):
        """
        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The energy density.
        """
        C = np.matmul(F.T, F)
        E = (C - np.eye(3)) / 2
        return 0.5 * lambda_in * np.trace(E) ** 2 + mu_in * np.tensordot(E, E)

    @staticmethod
    def hessian(F, lambda_in, mu_in):
        """
        Compute the Hessian of the energy density.

        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The Hessian of the energy density.
        """
        U, Sigma, Vt = svd_rotation_variant(F)
        I2 = np.sum(Sigma**2)
        front = -mu_in + lambda_in * 0.5 * (I2 - 3.0)
        s_sq = Sigma**2
        
        s_sq = Sigma ** 2
        s0s1 = Sigma[0] * Sigma[1]
        s0s2 = Sigma[0] * Sigma[2]
        s1s2 = Sigma[1] * Sigma[2]

        # Twist and flip modes
        eigenvalues = np.empty(9)
        # s_products = np.array([s1s2, s0s2, s0s1])  
        eigenvalues[:3] = [front + mu_in * (s_sq[1] + s_sq[2] - s1s2),
                   front + mu_in * (s_sq[0] + s_sq[2] - s0s2),
                   front + mu_in * (s_sq[0] + s_sq[1] - s0s1)]

        eigenvalues[3:6] = [front + mu_in * (s_sq[1] + s_sq[2] + s1s2),
                            front + mu_in * (s_sq[0] + s_sq[2] + s0s2),
                            front + mu_in * (s_sq[0] + s_sq[1] + s0s1)]

       # Scaling mode matrix
        A = np.diag(front + (lambda_in + 3.0 * mu_in) * s_sq)
        Sigma_outer = np.outer(Sigma, Sigma)
        i_upper = np.triu_indices(3, 1)
        A[i_upper] = lambda_in * Sigma_outer[i_upper]
        A.T[i_upper] = A[i_upper]  # Fill the lower triangle

        # Scaling modes
        Lambda, Q = np.linalg.eig(A)
        eigenvalues[6:] = Lambda

        # Compute the eigenvectors
        eigenvectors = np.zeros((9, 9))
        eigenvectors[:, :6] = build_twist_and_flip_eigenvectors(U, Vt)
        eigenvectors[:, 6:] = build_scaling_eigenvectors(U, Q, Vt)

        # 将特征值限制为非负值
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # 计算 Hessian 矩阵
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T



class COR:
    """
    Corotational linear elastic model.
    """

    @staticmethod
    def pk1_stress(F, lambda_in, mu_in):
        """
        Compute stress tensors.

        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The first Piola-Kirchhoff stress tensor
        """
        R, S = M3.polar_decomposition_array(F)
        return (lambda_in * np.trace(S - np.eye(3))) * R + (2 * mu_in) * (F - R)

    @staticmethod
    def energy_density(F, lambda_in, mu_in):
        """
        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The energy density.
        """
        R, S = M3.polar_decomposition_array(F)
        #  mu ||F - R||_F^2 + (lambda/2) tr^2 (R^T F - I)
        return mu_in * np.square(np.linalg.norm(F - R, ord="fro")) + (
            lambda_in / 2
        ) * np.square(np.trace(S - np.eye(3)))
