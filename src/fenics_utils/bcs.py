from dolfin.function.constant import Constant
from dolfin.cpp.math import near
from dolfin.cpp.common import DOLFIN_EPS
from dolfin.fem.dirichletbc import DirichletBC

from fenics_utils.mesh import get_mesh_axis_lims
from fenics_utils.mesh import get_mesh_axis_lim
from fenics_utils.mesh import get_mesh_axes_lims


# ???: will this break in parallel? (while getting info from partitioned mesh)


def set_dirichlet_bc(V, x_coord, value=0., axis=0, tol=DOLFIN_EPS):

    if type(value) in [int, float]:  # ???: what if np.float?
        value = Constant(value)

    def boundary_xmin(x, on_boundary):
        return on_boundary and near(x[axis], x_coord, tol)

    return DirichletBC(V, value, boundary_xmin)


def set_dirichlet_bcs_lims(V, value=0., axis=0, tol=DOLFIN_EPS):
    return [set_dirichlet_bc(V, x, value, axis, tol) for x in get_mesh_axis_lims(V.mesh(), axis=axis)]


def set_dirichlet_bc_lim(V, value=0., axis=0, right=True, tol=DOLFIN_EPS):
    x_coord = get_mesh_axis_lim(V.mesh(), axis=axis, right=right)
    return set_dirichlet_bc(V, x_coord, value, axis, tol)


def set_dirichlet_bcs_lims_all(V, value=0., tol=DOLFIN_EPS):
    bcs = []
    mins, maxs = get_mesh_axes_lims(V.mesh())
    for axis, (x_min, x_max) in enumerate(zip(mins, maxs)):
        bcs.append(set_dirichlet_bc(V, x_min, value, axis=axis, tol=tol))
        bcs.append(set_dirichlet_bc(V, x_max, value, axis=axis, tol=tol))

    return bcs
