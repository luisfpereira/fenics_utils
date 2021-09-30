from dolfin.fem.problem import LinearVariationalProblem
from dolfin.cpp.fem import LinearVariationalSolver

from fenics_utils.formulation.cfd import IncompressibleNsIpcs
from fenics_utils.formulation.cfd import AdvectionDiffusionScalar
from fenics_utils.solvers.linear import LinearSolver


class AdvectionDiffusionScalarNS:

    def __init__(self, V, Q, D, dt, mu, rho, f_ns, eps, f_ad, bcu, bcp, bcc=(),
                 solvers_parameters=None):
        self.V = V
        self.Q = Q
        self.D = D
        self.dt = dt
        self.mu = mu
        self.rho = rho
        self.f_ns = f_ns
        self.eps = eps
        self.f_ad = f_ad
        self.bcu = bcu
        self.bcp = bcp
        self.bcc = bcc
        self.solvers_parameters = solvers_parameters or self._set_default_solvers_parameters()

        # initialize empty variables
        self.ns_formulation = None
        self.ad_formulation = None

    def _set_default_solvers_parameters(self):
        return [{'linear_solver': 'bicgstab', 'preconditioner': 'hypre_amg'},
                {'linear_solver': 'bicgstab', 'preconditioner': 'hypre_amg'},
                {'linear_solver': 'cg', 'preconditioner': 'sor'},
                {'linear_solver': 'bicgstab', 'preconditioner': 'hypre_amg'}]

    def set(self):

        (a1, L1), (a2, L2), (a3, L3), (a4, L4) = self.set_eqs()
        u, _, p, _ = self.ns_formulation.get_functions()
        c, _ = self.ad_formulation.get_functions()

        solvers = [
            LinearSolver(a1, L1, u, self.bcu),
            LinearSolver(a2, L2, p, self.bcp),
            LinearSolver(a3, L3, u, None),
        ]
        problem_ad = LinearVariationalProblem(a4, L4, c, self.bcc)
        solvers.append(LinearVariationalSolver(problem_ad))

        # update solver parameters
        for solver, solver_parameters in zip(solvers, self.solvers_parameters):
            solver.parameters.update(solver_parameters)

        return solvers

    def set_eqs(self):
        """Set equations for all the variational problems.

        Useful if the user wants only the equations.
        """
        # NS formulation
        self.ns_formulation = IncompressibleNsIpcs(self.V, self.Q, self.dt,
                                                   self.mu, self.rho, self.f_ns)

        a1, L1 = self.ns_formulation.formulate_step1()
        a2, L2 = self.ns_formulation.formulate_step2()
        a3, L3 = self.ns_formulation.formulate_step3()
        u, _, p, _ = self.ns_formulation.get_functions()

        # advection-diffusion formulation
        self.ad_formulation = AdvectionDiffusionScalar(self.D, self.dt, self.eps, u,
                                                       self.f_ad)
        a4, L4 = self.ad_formulation.formulate()

        return [(a1, L1), (a2, L2), (a3, L3), (a4, L4)]

    def get_functions(self):
        if self.ns_formulation is None or self.ad_formulation is None:
            return None

        return self.ns_formulation.get_functions() + self.ad_formulation.get_functions()
