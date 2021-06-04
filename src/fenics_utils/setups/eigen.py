'''Defines simple eigenproblems (with very low flexibility) that are used to
test and develop code'''

from dolfin.function.functionspace import FunctionSpace
from dolfin.function.argument import TrialFunction
from dolfin.function.argument import TestFunction

from ufl import dot
from ufl import grad
from ufl import dx

from fenics_utils.mesh import create_unit_hypercube


def set_basic(mesh, V=None):
    return _formulate_basic(mesh, V)


def set_basic_unit_hypercube(n, V=None):
    '''
    Args:
        n (array-like)
    '''
    return _formulate_basic(create_unit_hypercube(*n), V)


def _formulate_basic(mesh, V=None):

    if V is None:
        V = FunctionSpace(mesh, 'CG', 1)

    u_h = TrialFunction(V)
    v = TestFunction(V)

    a = dot(grad(u_h), grad(v)) * dx

    return a
