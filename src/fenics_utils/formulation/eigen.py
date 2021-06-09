from dolfin.function.functionspace import FunctionSpace
from dolfin.function.argument import TrialFunction
from dolfin.function.argument import TestFunction

from ufl import inner
from ufl import grad
from ufl import dx


def formulate_laplacian(mesh, V=None):

    if V is None:
        V = FunctionSpace(mesh, 'CG', 1)

    u_h = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u_h), grad(v)) * dx
    b = inner(u_h, v) * dx
    dummy = inner(1., v) * dx

    return V, a, b, dummy
