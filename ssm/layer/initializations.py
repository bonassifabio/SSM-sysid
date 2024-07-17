from collections import namedtuple
from typing import Protocol, runtime_checkable

import numpy as np
import torch

FullLTI = namedtuple('FullLTI', ['A', 'B', 'C', 'D'])
DiagonalLTI = namedtuple('DiagonalLTI', ['Λ', 'Bd', 'Cd', 'Dd'])

def project_into(lti: FullLTI, V: np.ndarray) -> DiagonalLTI:
    """Project the LTI (A, B, C, D) into the eigenspace described by V

    Args:
        lti (FullLTI): The full LTI system (A, B, C, D)
        V (np.ndarray): The eigenspace

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The diagonalized system matrices (Λ, B_tilde, C_tilde)
    """
    Λ = np.diag(V.conj().T @ lti.A @ V)
    Bd = V.conj().T @ lti.B
    Cd = lti.C @ V
    return DiagonalLTI(Λ, Bd, Cd, lti.D)


def project_back(diagonal: DiagonalLTI, V: np.ndarray, force_real: bool = False) -> FullLTI:
    """Project the diagonalized LTI system (Λd, Bd, Cd) back from the eigenspace described by V.

    Args:
        diagonal (DiagonalLTI): The diagonalized LTI system (Λ, Bd, Cd, D) 
        V (np.ndarray): The eigenspace
        force_real (bool, optional): Whether to force the returned matrices to be real-valued. Defaults to False.

    Returns:
        FullLTI: The original LTI system matrices (A, B, C, D)
    """
    if diagonal.Λ.shape[0] != V.shape[0]:
        diagonal_ = expand_complex_conjugates(diagonal)
    else: 
        diagonal_ = diagonal

    Λ = np.diag(diagonal_.Λ) if diagonal_.Λ.shape[0] > 1 else diagonal_.Λ

    A = V @ Λ @ V.conj().T
    B = V @ diagonal_.Bd
    C = diagonal_.Cd @ V.conj().T
    D = diagonal_.Dd

    if force_real:
        def negligible_imag(x):
            return np.all(np.abs(x.imag) < 1e-6)
        
        assert negligible_imag(A) and negligible_imag(B) and negligible_imag(C) and negligible_imag(D)

        A = A.real
        B = B.real
        C = C.real
        D = D.real
    
    return FullLTI(A, B, C, D)


def sort_complex_conjugates(diagonal: DiagonalLTI, V: np.ndarray) -> tuple[DiagonalLTI, np.ndarray, np.ndarray]:
    """Sort the eigenvalues of the diagonalized LTI system (Λ, Bd, Cd). The sorting criterion is ascending (negative) real part, then ascending (non-negative) imaginary part.

    Args:
        diagonal (DiagonalLTI): The diagonalized LTI system (Λ, Bd, Cd, D)
        V (np.ndarray): The eigenspace (V)

    Returns:
        tuple[DiagonalLTI, np.ndarray]: The sorted diagonal LTI system, and the sort eigenspace
    """
    # For the moment our assumption is that the number of states is even.
    if diagonal.Λ.shape[0] % 2 != 0:
        raise ValueError('Λ must have an even number of elements!')
    
    # Sorting criterion: first, all eigenvalues with positive real part in ascending order. 
    # Then, their complex conjugates in the appropriate order.
    has_positive_imag = -(diagonal.Λ.imag >= 0.0).astype('f2')
    re = diagonal.Λ.real.astype('f2')
    imag_nabs = np.abs(diagonal.Λ.imag).astype('f2')
    sort_criterion = np.array(list(zip(has_positive_imag, re, imag_nabs)), dtype=[('pos-imag', 'f2'), ('real', 'f2'), ('imag', 'f2')])

    # Sort by descending imaginary part
    idx_sort = np.argsort(sort_criterion, axis=0, order=['pos-imag', 'real', 'imag'])
    sorted_diagonal = DiagonalLTI(diagonal.Λ[idx_sort], diagonal.Bd[idx_sort, :], diagonal.Cd[:, idx_sort], diagonal.Dd)
    sorted_V = V[:, idx_sort][idx_sort, :]

    return sorted_diagonal, sorted_V, idx_sort


def trim_complex_conjugates(diagonal: DiagonalLTI, V: np.ndarray) -> DiagonalLTI:
    """Remove complex conjugate eigenvalues. Keep only those with non-negative imaginary part.

    Args:
        diagonal (DiagonalLTI): The diagonalized LTI system (Λ, Bd, Cd, D) with sorted eigenvalues.

    Returns:
        tuple[DiagonalLTI]: The trimmed diagonalized LTI system.
    """
    # Check that the states have been properly ordered
    N = diagonal.Λ.shape[0]
    if N % 2 != 0:
        raise ValueError('Λ must have an even number of elements!')
    
    assert np.allclose(diagonal.Λ[:N // 2], diagonal.Λ[N // 2:].conj())
    assert np.allclose(diagonal.Bd[:N // 2, :], diagonal.Bd[N // 2:, :].conj())
    assert np.allclose(diagonal.Cd[:, :N // 2], diagonal.Cd[:, N // 2:].conj())
    assert np.allclose(V[:, :N // 2], V[:, N // 2:].conj())

    # Trim the complex conjugates
    return DiagonalLTI(diagonal.Λ[:N // 2], diagonal.Bd[:N // 2, :], diagonal.Cd[:, :N // 2], diagonal.Dd)


def expand_complex_conjugates(diagonal: DiagonalLTI) -> DiagonalLTI:
    """Expand complex conjugate eigenvalues with the complex-conjugate eigenvalues.

    Args:
        diagonal (DiagonalLTI): The diagonalized LTI system (Λ, Bd, Cd, D) with sorted eigenvalues and trimmed complex conjugates.

    Returns:
        DiagonalLTI: The expanded diagonalized LTI system.
    """
    Λ = np.concatenate((diagonal.Λ, diagonal.Λ.conj()))
    Bd = np.concatenate((diagonal.Bd, diagonal.Bd.conj()), axis=0)
    Cd = np.concatenate((diagonal.Cd, diagonal.Cd.conj()), axis=1)
    return DiagonalLTI(Λ, Bd, Cd, diagonal.Dd)


def match_mat_init(F_in: int, F_out: int, init: str, scale: float = 1.0, complex: bool = False) -> np.ndarray:
    """Match the initialization function to the correct function signature.

    Args:
        F_in (int): The number of input features.
        F_out (int): The number of output features.
        init (str): The initialization function name.
        scale (float, optional): Scaling factor for the initialization. Defaults to 1.0.
        complex (bool, optional): Whether the matrix is complex or not. Defaults to False.
    
    Returns:
        np.ndarray: The initialized matrix.
    """
    F_out_ = 2 * F_out if complex else F_out

    W = torch.zeros(F_out_, F_in)
    match init:
        case 'xavier-normal':
            torch.nn.init.xavier_normal_(W)
        case 'xavier-uniform':
            torch.nn.init.xavier_uniform_(W)
        case 'kaiming-normal':
            torch.nn.init.kaiming_normal_(W)
        case 'kaiming-uniform':
            torch.nn.init.kaiming_uniform_(W)
        case 'orthogonal':
            torch.nn.init.orthogonal_(W)
        case 'he-normal':
            torch.nn.init.he_normal_(W)
        case 'ones':
            torch.nn.init.ones_(W)

    if complex:
        W = W[:F_out, :] + 1j * W[F_out:, :]

    return scale * W.detach().numpy()

def make_HiPPO_LegN(N: int, in_features: int = 1, out_features: int = 1):
    """Compute the HiPPO-LegS-N matrix"""

    def a_nk(n, k):
        # Compute the (n, k) element of the HiPPO-LegS-N matrix
        return - np.sign(n - k) * np.sqrt(n + 0.5) * np.sqrt(k + 0.5) if n != k else -0.5

    # We first compute the full HiPPO-LegS-N matrix, and then throw away half the eigenvalues
    A_N = np.zeros((N, N))
    B_N = np.sqrt(2 * np.arange(N) + 1.0).reshape((N, 1)).repeat(in_features, axis=1)

    # Naively construct HiPPO-LegS-N matrix
    for n in np.arange(N):
        for k in np.arange(N):
            A_N[n, k] = a_nk(n, k)

    return A_N, B_N


def jordan_transformation(N: int) -> np.ndarray:
    """Compute the Jordan transformation matrix for the very specific case where the diagonal system has block-conjugate eigenvalues.
       That is, Λ = blkdiag(λ, λ.conj()). This transformation matrix project this complex-valued diagonal system to an equivalent (block-diagonal) real-valued system.
    """
    # First, we expand the complex conjugates
    Nh = N // 2

    # Change of variable to the Block-Jordan canonical form
    T = np.zeros((2 * Nh, 2 * Nh)).astype('complex128')
    T[:Nh, :Nh] = 1.0 / np.sqrt(2) * np.eye(Nh)
    T[Nh:, :Nh] = 1.0j / np.sqrt(2) * np.eye(Nh)
    T[:, Nh:] = T[:, :Nh].conj()

    return T


@runtime_checkable
class S5Initializer(Protocol):

    def __call__(self, N: int, in_features: int, out_features: int) -> tuple[DiagonalLTI, np.ndarray, np.ndarray]:
        """Protocol class defining the interface for S5 initializers.

        Args:
            N (int): The number of states.
            in_features (int): The number of input features.
            out_features (int): The number of output features.

        Returns:
            tuple[DiagonalLTI, np.ndarray, np.ndarray]: The Diagonal State-Space matrices, the initial value of the timescale parameter, and the projection operator.
        """
        ...


class DummyInitializer(S5Initializer):

    def __call__(self, N: int, in_features: int, out_features: int) -> tuple[DiagonalLTI, np.ndarray, np.ndarray]:
        if N % 2 != 0:
            raise ValueError('N must be even. The odd case has not been implemented!')
        
        Λ =  -1 * np.ones((N // 2,)) + 1j * (1.0 + np.arange(N // 2))
        Bd = (1.0 + np.arange(N // 2)).reshape((N // 2, 1)).repeat(in_features, axis=1)
        Cd = (1.0 + np.arange(N // 2)).reshape((1, N // 2)).repeat(out_features, axis=0)
        V = np.eye(N // 2)

        diag_sys = DiagonalLTI(Λ, Bd, Cd, np.zeros((out_features, in_features)))
        return diag_sys, V
    

class HippoDiagonalizedInitializer(S5Initializer):

    def __init__(self, scale: float = 1.0, c_init_fcn: str = 'xavier-normal') -> None:
        """HiPPO-LegS-Normal Diagonalized Initializer

        Args:
            scale (float, optional): Scaling factor for the HiPPO-LegS-N matrix. Defaults to 1.0.
            c_init_fcn (str, optional): Initialization for the C matrix. Defaults to 'xavier-normal'.
        """
        self.scale = scale
        self.c_init_fcn = c_init_fcn
        
    def __call__(self, N: int, in_features: int, out_features: int) -> tuple[DiagonalLTI, np.ndarray]:
        if N % 2 != 0:
            raise ValueError('N must be even. The odd case has not been implemented!')

        # Compute the Hippo-LegS-N matrix
        A_N, B_N = make_HiPPO_LegN(N, in_features, out_features)
        C_N = match_mat_init(F_in=N, F_out=out_features, init=self.c_init_fcn, complex=False)
        fullsys = FullLTI(A_N * self.scale, B_N * self.scale, C_N, np.zeros((out_features, in_features)))

        # We now compute the eigenvalues and eigenvectors of the HiPPO-LegS-N matrix
        _, V = np.linalg.eig(A_N)
        # Project the HiPPO-LegS-N matrix into the eigenspace
        diagonal = project_into(fullsys, V)
        
        # Sort the eigenvalues and remove the complex conjugates
        diagonal, V, _ = sort_complex_conjugates(diagonal, V)
        diagonal = trim_complex_conjugates(diagonal, V)

        return diagonal, V


class DiagonalInitializer(S5Initializer):

    def __init__(self, dt: float, scale: float = 1.0, phase: float = 45.0, init_fcn: str = 'xavier-normal', complex: bool = True, unity_gain: bool = True):
        """Diagonal Initializer
        
        Args:
            dt (float): The sampling time.
            scale (float, optional): Scaling factor for the eigenvalues. Defaults to 1.0.
            phase (float, optional): Phase of the eigenvalues. Defaults to 45.0.
            init_fcn (str, optional): Initialization for the B and C matrices. Defaults to 'xavier-normal'.
            complex (bool, optional): Whether the B and C matrices are complex or not. Defaults to True.
        """ 
        self.ω_max = np.pi / dt     # Maximum frequequency (rad/s) content according to Shannon-Nyquist theorem
        self.ω_min = 0.0
        self.scale = scale
        self.init_fcn = init_fcn
        self.complex = complex
        self.phase = phase
        self.enforce_unity_gain = unity_gain

        assert phase >= 0.0 and phase <= 90.0
        
    def __call__(self, N: int, in_features: int, out_features: int) -> tuple[DiagonalLTI, np.ndarray]:
        if N % 2 != 0:
            raise ValueError('N must be even. The odd case has not been implemented!')

        # Modulus of the eigenvalues
        dω = (self.ω_max - self.ω_min) / (N // 2)
        mag = np.linspace(self.ω_min + dω, self.ω_max, N // 2) * self.scale
        Λ = mag * np.exp(1j * np.deg2rad(180 - self.phase))
        Bd = match_mat_init(F_in=in_features, F_out=N // 2, init=self.init_fcn, complex=self.complex)
        Cd = match_mat_init(F_in=N // 2, F_out=out_features, init=self.init_fcn, complex=self.complex)
        Dd = np.zeros((out_features, in_features))
        V = jordan_transformation(N)

        # Ensure that approximately-unitary static gain for the diagonalized system
        if self.enforce_unity_gain:
            Bd = mag.reshape(-1 ,1) * Bd

        return DiagonalLTI(Λ, Bd, Cd, Dd), V
