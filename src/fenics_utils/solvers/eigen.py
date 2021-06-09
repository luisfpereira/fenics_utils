import scipy

from dolfin.cpp.la import SLEPcEigenSolver

from fenics_utils.solvers.utils import collect_SLEPc_eigenpairs


class MySLEPcEigenSolver(SLEPcEigenSolver):

    def __init__(self, A, B, bcs=None, alpha=1e6, solver='krylov-schur',
                 spectrum='smallest magnitude', problem_type='gen_hermitian'):
        # TODO: deal with bcs

        super().__init__(A, B)
        self.parameters['solver'] = solver
        self.parameters['spectrum'] = spectrum
        self.parameters['problem_type'] = problem_type

    def solve(self, n_eig=5):
        super().solve(n_eig)

        return collect_SLEPc_eigenpairs(self)


class ScipySparseEigenSolver:
    '''
    Solves generalized eigenproblem for symmetric matrices.

    Notes:
        Looks to emulate behavior of SLEPcEigenSolver (goal is to use both 
    indistinctively).
    '''

    def __init__(self, A, B, bcs=None, alpha=1e6, spectrum='smallest magnitude'):
        # TODO: deal with bcs and alpha
        self.A = self._get_scipy_sparse(A)
        self.B = self._get_scipy_sparse(B)
        self.params = {'spectrum': spectrum}

    def _map_spectrum(self):
        map_dict = {'smallest magnitude': 'SM',
                    'largest magnitude': 'LM', }

        return {'which': map_dict[self.params['spectrum']]}

    def _get_scipy_sparse(self, M):
        return scipy.sparse.csr_matrix(M.mat().getValuesCSR()[::-1])

    def solve(self, n_eig=5):
        return scipy.sparse.linalg.eigsh(self.A, M=self.B, k=n_eig,
                                         **self._map_spectrum())
