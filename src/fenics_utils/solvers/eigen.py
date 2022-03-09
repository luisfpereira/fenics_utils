import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from dolfin.cpp.la import SLEPcEigenSolver
from dolfin.cpp.common import DOLFIN_EPS


class MySLEPcEigenSolver(SLEPcEigenSolver):

    def __init__(self, A, B, solver='generalized-davidson',
                 spectrum='smallest magnitude', problem_type='gen_hermitian',
                 comm=None):

        if comm is None:
            super().__init__(A, B)
        else:
            super().__init__(comm)
            self.set_operators(A, B)

        self.parameters['solver'] = solver
        self.parameters['spectrum'] = spectrum
        self.parameters['problem_type'] = problem_type
        self.parameters['tolerance'] = DOLFIN_EPS

    def solve(self, n_eig=5):
        super().solve(n_eig)

        return collect_SLEPc_eigenpairs(self)


class ScipySparseEigenSolver:
    """Solves generalized eigenproblem for symmetric matrices.

    Notes:
        Looks to emulate behavior of SLEPcEigenSolver (goal is to use both \
    indistinctively).
    """

    def __init__(self, A, B, spectrum='smallest magnitude'):
        self.A = self._get_scipy_sparse(A)
        self.B = self._get_scipy_sparse(B)
        self.params = {'spectrum': spectrum}

    def _map_spectrum(self):
        map_dict = {'smallest magnitude': 'SM',
                    'largest magnitude': 'LM', }

        return {'which': map_dict[self.params['spectrum']]}

    def _get_scipy_sparse(self, M):
        return sparse.csr_matrix(M.mat().getValuesCSR()[::-1])

    def solve(self, n_eig=5):
        return linalg.eigsh(self.A, M=self.B, k=n_eig,
                            **self._map_spectrum())


def collect_SLEPc_eigenpairs(solver):
    """Returns only real part.
    """

    w, v = [], []
    for i in range(solver.get_number_converged()):
        r, _, rv, _ = solver.get_eigenpair(i)
        w.append(r)
        v.append(rv)

    return np.array(w), np.array(v).T
