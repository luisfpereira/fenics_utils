'''
Defines simple examples with heat equation (with very low flexibility) that
are used to develop and test code.
'''

from dolfin.cpp.generation import UnitSquareMesh

from dolfin.function.functionspace import FunctionSpace
from dolfin.function.expression import Expression
from dolfin.function.constant import Constant
from dolfin.function.function import Function

from fenics_utils.mesh import get_mesh_axis_lims
from fenics_utils.bcs import set_dirichlet_bc
from fenics_utils.bcs import set_dirichlet_bc_lim
from fenics_utils.formulation.heat import get_formulation_constant_props


def set_linear_dirichlet_constant(mesh, dt, var_name='Temperature',
                                  bc_temperature=1300., temperature=300.,
                                  source_value=0., axis=1, right=True, tol=1e-4,
                                  conductivity=1., density=1., specific_heat=1.):
    '''
    Heat equation problem where an edge/surface is set at a given temperature
    and everything else at another.

    Parameters
    ----------
    axis : int
        Controls boundary to which Dirichlet bc is applied.
    right : bool
        Dirichlet bc is applied to the planar and perpendicular face found in
        the max (True) or min (False) coordinate.
    '''

    # function space
    V = FunctionSpace(mesh, 'Lagrange', 1)

    # boundary conditions
    bcs = [set_dirichlet_bc_lim(V, value=bc_temperature, axis=axis, right=right,
                                tol=tol)]

    # initial condition
    u_initial = Constant(temperature)
    u_n = Function(V, name=var_name)
    u_n.interpolate(u_initial)

    # problem formulation
    f = Constant(source_value)
    u_n, a, L = get_formulation_constant_props(
        V, u_initial, f, dt, conductivity, density, specific_heat,
        var_name=var_name)

    u = Function(V, name=var_name)

    return u_n, a, L, u, bcs


def set_linear_equal_opposite(mesh, dt, var_name='Temperature', axis=0,
                              bc_temperature=300., source_value=0.,
                              conductivity=1., density=1., specific_heat=1.,
                              V=None):
    '''
    Heat equation problem where two opposite edges/faces are set at the same
    temperature and the points in between follow a quadratic variation.

    Notes
    -----
    * assumes opposite faces of interest are planar and perpendicular to the
    axis.
    '''

    # function space
    if V is None:
        V = FunctionSpace(mesh, 'Lagrange', 1)

    # boundary conditions
    x_min, x_max = get_mesh_axis_lims(mesh, axis)
    bcs = [set_dirichlet_bc(V, x, bc_temperature, axis) for x in [x_min, x_max]]

    # initial condition
    c = (x_max - x_min) / 2
    param = bc_temperature / (c**2)
    u_initial = Expression('param * (x[axis] - c) * (x[axis] - c)',
                           degree=2, param=param, c=c, name='T', axis=axis)

    # problem formulation
    f = Constant(source_value)
    u_n, a, L = get_formulation_constant_props(
        V, u_initial, f, dt, conductivity, density, specific_heat,
        var_name=var_name)

    u = Function(V, name=var_name)

    return u_n, a, L, u, bcs


def set_heat_eq_2d(dt, n=8):
    '''
    Basic example where only the mesh refinement is controlled.
    '''
    # TODO extend with hypercube

    var_name = 'Temperature'

    # mesh
    mesh = UnitSquareMesh(n, n)

    return set_linear_equal_opposite(mesh, dt, var_name=var_name)
