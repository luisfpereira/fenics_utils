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


# TODO: add material parameters to all the functions


def set_heat_dirichlet_constant(mesh, dt, var_name='Temperature',
                                bc_temperature=1300., temperature=300.,
                                source_value=0., axis=1, face='max', tol=1e-4):
    '''
    Heat equation problem where an edge/surface is set at a given temperature
    and everything else at another.

    Parameters
    ----------
    axis : int
        Controls boundary to which Dirichlet bc is applied.
    face : str
        Dirichlet bc is applied to the planar and perpendicular face found in
        the max ('max') or min ('min') coordinate.
    '''

    # function space
    V = FunctionSpace(mesh, 'Lagrange', 1)

    # boundary conditions (only upper hole has Dirichlet bc)
    mesh_coords = mesh.coordinates()
    x_bc = mesh_coords[:, axis].max() if face == 'max' else mesh_coords[:, axis].min()
    u_xbc = Constant(bc_temperature)

    def boundary_x(x, on_boundary):
        return on_boundary and near(x[axis], x_bc, tol)

    bcs = [DirichletBC(V, u_xbc, boundary_x)]

    # initial condition
    u_initial = Constant(temperature)
    u_n = Function(V, name=var_name)
    u_n.interpolate(u_initial)

    # problem formulation
    u_h = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(source_value)

    a = u_h * v * dx + dt * dot(grad(u_h), grad(v)) * dx
    L = (u_n + dt * f) * v * dx

    u = Function(V, name=var_name)

    return u_n, a, L, u, bcs


def set_heat_equal_opposite(mesh, dt, var_name='Temperature', axis=0,
                            bc_temperature=300., source_value=0.):
    '''
    Heat equation problem where two opposite edges/faces are set at the same
    temperature and the points in between follow a quadratic variation.

    Notes
    -----
    * assumes opposite faces of interest are planar and perpendicular to the
    axis.
    '''

    # function space
    V = FunctionSpace(mesh, 'Lagrange', 1)

    # boundary conditions
    mesh_coords = mesh.coordinates()
    x_min, x_max = mesh_coords[:, axis].min(), mesh_coords[:, axis].max()
    u_xmin, u_xmax = Constant(bc_temperature), Constant(bc_temperature)

    def boundary_xmin(x, on_boundary):
        return on_boundary and near(x[axis], x_min)

    def boundary_xmax(x, on_boundary):
        return on_boundary and near(x[axis], x_max)

    bcs = [DirichletBC(V, u_xmin, boundary_xmin),
           DirichletBC(V, u_xmax, boundary_xmax)]

    # initial condition
    c = (x_max - x_min) / 2
    param = u_xmin.values()[0] / (c**2)
    u_initial = Expression('param * (x[axis] - c) * (x[axis] - c)',
                           degree=2, param=param, c=c, name='T', axis=axis)

    u_n = Function(V, name=var_name)
    u_n.interpolate(u_initial)

    # problem formulation
    u_h = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(source_value)

    a = u_h * v * dx + dt * dot(grad(u_h), grad(v)) * dx
    L = (u_n + dt * f) * v * dx

    u = Function(V, name=var_name)

    return u_n, a, L, u, bcs


def set_heat_eq_2d(dt, n=8):
    '''
    Basic example where only the mesh refinement is controlled.
    '''

    var_name = 'Temperature'

    # mesh
    mesh = UnitSquareMesh(n, n)

    return set_heat_equal_opposite(mesh, dt, var_name=var_name)
