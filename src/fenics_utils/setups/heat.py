'''
Defines simple examples with heat equation (with very low flexibility) that
are used to develop and test code.
'''

from dolfin.cpp.generation import UnitSquareMesh
from dolfin.cpp.math import near

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


def set_heat_eq_2d(dt, n=8):
    '''
    Basic example where only the mesh refinement is controlled.
    '''

    var_name = 'Temperature'

    # mesh
    mesh = UnitSquareMesh(n, n)

    # function space
    V = FunctionSpace(mesh, 'P', 1)

    # bcs
    x_min, x_max = 0., 1.
    u_xmin, u_xmax = Constant(500.), Constant(500.)

    def boundary_xmin(x, on_boundary):
        return on_boundary and near(x[0], x_min)

    def boundary_xmax(x, on_boundary):
        return on_boundary and near(x[0], x_max)

    bcs = [DirichletBC(V, u_xmin, boundary_xmin),
           DirichletBC(V, u_xmax, boundary_xmax)]

    # initial condition
    param = u_xmin.values()[0] / (.5**2)
    c = (x_max - x_min) / 2.

    u_initial = Expression('param * (x[0] - c) * (x[0] - c)',
                           degree=2, param=param, c=c, name='T')

    u_n = Function(V, name=var_name)
    u_n.interpolate(u_initial)

    # problem formulation
    u_h = TrialFunction(V)
    v = TestFunction(V)

    f = Constant(0.)
    a = u_h * v * dx + dt * dot(grad(u_h), grad(v)) * dx
    L = (u_n + dt * f) * v * dx

    u = Function(V, name=var_name)

    return u_n, a, L, u, bcs
