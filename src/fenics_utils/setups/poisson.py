'''
Defines simple examples with Poisson equation (with very low flexibility) that
are used to develop and test code.
'''


from dolfin.cpp.generation import UnitSquareMesh

from dolfin.function.functionspace import FunctionSpace
from dolfin.function.expression import Expression
from dolfin.function.argument import TrialFunction
from dolfin.function.argument import TestFunction
from dolfin.function.constant import Constant
from dolfin.function.function import Function

from dolfin.fem.dirichletbc import DirichletBC


from ufl import dot
from ufl import dx
from ufl import grad


def set_poisson_eq_2d(n=8):
    """Basic example where only the mesh refinement is controlled.
    """

    # mesh
    mesh = UnitSquareMesh(n, n)

    # function space
    V = FunctionSpace(mesh, 'P', 1)

    # bcs
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # variational problem
    # TODO: create formulation for this problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx

    # define unknown
    u = Function(V)

    return a, L, u, bc
