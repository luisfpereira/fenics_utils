import dolfin.cpp as cpp
from dolfin.fem.assembling import assemble


class LinearSolver:
    """A linear solver that assembles A only once.
    """

    def __init__(self, a, L, u, bcs=None, method='default',
                 preconditioner='default'):
        self.a = a
        self.L = L
        self.u = u
        self.bcs = bcs if bcs is not None else ()
        self.parameters = {'linear_solver': method,
                           'preconditioner': preconditioner}

        # assemble A
        self._assemble_lhs()

    def _assemble_lhs(self):
        self.A = assemble(self.a)
        for bc in self.bcs:
            bc.apply(self.A)

    def _assemble_rhs(self):
        b = assemble(self.L)
        for bc in self.bcs:
            bc.apply(b)

        return b

    def solve(self):
        b = self._assemble_rhs()

        return cpp.la.solve(self.A, self.u.vector(), b,
                            self.parameters['linear_solver'],
                            self.parameters['preconditioner'])
