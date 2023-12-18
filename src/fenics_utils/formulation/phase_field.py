
from dolfin.function.function import Function
from dolfin.function.argument import TrialFunction
from dolfin.function.argument import TestFunction

from ufl import sym
from ufl import grad
from ufl import tr
from ufl import Identity
from ulf import inner
from ufl import dx
from ufl import dev
from ufl import conditional
from ufl import lt
from ufl import lhs
from ufl import rhs


# TODO: set references to this (Paneda)


def epsilon(u):
    return sym(grad(u))


def sigma(u, lmbda, mu):
    return 2.0 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(len(u))


def psi(u, lmbda, mu):
    return 0.5 * (lmbda + mu) * (0.5 * (tr(epsilon(u)) + abs(tr(epsilon(u)))))**2 +\
        mu * inner(dev(epsilon(u)), dev(epsilon(u)))


def H(u_n, u, H_n, lmbda, mu):
    return conditional(lt(psi(u_n, lmbda, mu), psi(u, lmbda, mu)), psi(u, lmbda, mu), H_n)


class ModifiedElasticity:

    def __init__(self, V, lmbda, mu, p_n, u_name='Displacement'):
        self.V = V
        self.u_name = u_name
        self.lmbda = lmbda
        self.mu = mu
        self.p_n = p_n
        self.sigma = lambda u: sigma(u, lmbda, mu)

        self._define_functions()

    def _define_functions(self):
        self.u_h = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        # functions for solution at previous and current time steps
        self.u = Function(self.V, name=self.u_name)
        self.u_n = Function(self.V, name=self.u_name)

    def get_functions(self):
        return self.u, self.u_n

    def formulate(self):
        F = ((1.0 - self.p_n)**2) * inner(grad(self.v), self.sigma(self.u_h)) * dx
        a = lhs(F)
        L = rhs(F)

        return a, L


class PhaseField:

    def __init__(self, V, u, u_n, p_name='Damage'):
        self.V = V
        self.p_name = p_name
        self.u = u
        self.u_n = u_n

        self._define_functions()

    def _define_functions(self):
        self.p_h = TrialFunction(self.V)
        self.q = TestFunction(self.V)

        # functions for solution at previous and current time steps
        self.p = Function(self.V, name=self.p_name)
        self.p_n = Function(self.V, name=self.p_name)

    def get_functions(self):
        return self.p, self.p_n

    def formulate(self):
        F = (Gc * l * inner(grad(self.p_h), grad(self.q)) + ((Gc / l) + 2.0 * H(uold, unew, Hold))
             * inner(p, q) - 2.0 * H(uold, unew, Hold) * q) * dx
