'''
Defines formulation of heat equation cases.
'''

from dolfin.function.argument import TrialFunction
from dolfin.function.argument import TestFunction
from dolfin.function.function import Function
from dolfin.function.constant import Constant

from ufl import dot
from ufl import dx
from ufl import grad


def get_formulation(V, u_initial, f, dt, var_name='Temperature'):

    # interpolate initial condition
    u_n = Function(V, name=var_name)
    u_n.interpolate(u_initial)

    # function spaces
    u_h = TrialFunction(V)
    v = TestFunction(V)

    # formulation
    a = u_h * v * dx + dt * dot(grad(u_h), grad(v)) * dx
    L = (u_n + dt * f) * v * dx

    return u_n, a, L


def get_formulation_constant_props(V, u_initial, f, dt, conductivity, density,
                                   specific_heat, var_name='Temperature'):

    # medium properties
    k = Constant(conductivity)
    rho = Constant(density)
    c_p = Constant(specific_heat)

    # interpolate initial condition
    u_n = Function(V, name=var_name)
    u_n.interpolate(u_initial)

    # function spaces
    u_h = TrialFunction(V)
    v = TestFunction(V)

    # formulation
    param1 = rho * c_p
    param2 = k / param1
    a = u_h * v * dx + param2 * dt * dot(grad(u_h), grad(v)) * dx
    L = (u_n + param1 * dt * f) * v * dx

    return u_n, a, L
