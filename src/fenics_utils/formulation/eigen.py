
from dolfin.function.argument import TrialFunction
from dolfin.function.argument import TestFunction

from ufl import inner
from ufl import grad
from ufl import dx


def formulate_laplacian(V):

    u_h = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u_h), grad(v)) * dx
    b = inner(u_h, v) * dx
    dummy = inner(1., v) * dx

    return a, b, dummy
