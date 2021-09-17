from dolfin.function.function import Function
from dolfin.function.argument import TrialFunction
from dolfin.function.argument import TestFunction
from dolfin.function.specialfunctions import FacetNormal

from ufl import sym
from ufl import nabla_grad
from ufl import Identity
from ufl import dot
from ufl import inner
from ufl import dx
from ufl import lhs
from ufl import rhs
from ufl import ds
from ufl import div
from ufl import grad


def epsilon(u):
    """Defines strain-rate tensor.
    """
    return sym(nabla_grad(u))


def sigma(u, p, mu):
    """Defines stress tensor.
    """
    return 2 * mu * epsilon(u) - p * Identity(len(u))


class IncompressibleNsIpcs:
    """Incompressible Navier-Stokes with IPCS splitting method [1]_.

    Args:
        V: Velocity function space.
        Q: Pressure function space.

    Notes:
        IPCS stands for Incremental Pressure Correction Scheme.

    References:
        .. [1] `The Navier–Stokes equations <https://fenicsproject.org/pub/tutorial/html/._ftut1009.html#ftut1:NS>`_.
    """

    def __init__(self, V, Q, dt, mu, rho, f, u_name='Velocity',
                 p_name='Pressure'):
        self.V = V
        self.Q = Q
        self.dt = dt
        self.mu = mu
        self.rho = rho
        self.u_name = u_name
        self.p_name = p_name
        self.f = f

        self._define_functions()
        self._define_aux_variables(V.mesh())

    def _define_functions(self):

        # trial and test functions
        self.u_h = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.p_h = TrialFunction(self.Q)
        self.q = TestFunction(self.Q)

        # functions for solutions at previous and current time steps
        self.u_n = Function(self.V, name=self.u_name)
        self.u = Function(self.V, name=self.u_name)
        self.p_n = Function(self.Q, name=self.p_name)
        self.p = Function(self.Q, name=self.p_name)

    def _define_aux_variables(self, mesh):
        self.U = 0.5 * (self.u_n + self.u_h)
        self.n = FacetNormal(mesh)

    def get_functions(self):
        return self.u, self.u_n, self.p, self.p_n

    def formulate_step1(self):
        """Defines variational problem for step 1 (tentative velocity).
        """
        F = self.rho * dot((self.u_h - self.u_n) / self.dt, self.v) * dx + \
            self.rho * dot(dot(self.u_n, nabla_grad(self.u_n)), self.v) * dx \
            + inner(sigma(self.U, self.p_n, self.mu), epsilon(self.v)) * dx \
            + dot(self.p_n * self.n, self.v) * ds \
            - dot(self.mu * nabla_grad(self.U) * self.n, self.v) * ds \
            - dot(self.f, self.v) * dx
        a = lhs(F)
        L = rhs(F)

        return a, L

    def formulate_step2(self):
        """Defines variational problem for step 2 (pressure with tentative velocity).
        """
        a = dot(nabla_grad(self.p_h), nabla_grad(self.q)) * dx
        L = dot(nabla_grad(self.p_n), nabla_grad(self.q)) * dx - (1 / self.dt) * div(self.u) * self.q * dx

        return a, L

    def formulate_step3(self):
        """Defines variational problem for step 3 (velocity).
        """
        a = dot(self.u_h, self.v) * dx
        L = dot(self.u, self.v) * dx - self.dt * dot(nabla_grad(self.p - self.p_n), self.v) * dx

        return a, L


class AdvectionDiffusionScalar:
    """Advection-diffusion scalar equation [1]_.

    Args:
        D: Concentration function space.

    References:
        .. [1] `A system of advection–diffusion–reaction equations <A system of advection–diffusion–reaction equations>`_.
    """

    def __init__(self, D, dt, eps, u, f, c_name='Concentration'):
        self.D = D
        self.c_name = c_name
        self.dt = dt
        self.eps = eps
        self.u = u
        self.f = f

        self._define_functions()

    def _define_functions(self):

        # trial and test functions
        self.c_h = TrialFunction(self.D)
        self.d = TestFunction(self.D)

        # functions for solution at previous and current time steps
        self.c = Function(self.D, name=self.c_name)
        self.c_n = Function(self.D, name=self.c_name)

    def get_functions(self):
        return self.c, self.c_n

    def formulate(self):
        a = self.c_h / self.dt * self.d * dx \
            + dot(self.u, grad(self.c_h)) * self.d * dx \
            + self.eps * dot(grad(self.c_h), grad(self.d)) * dx
        L = self.f * self.d * dx + self.c_n / self.dt * self.d * dx

        return a, L
