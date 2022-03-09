"""Defines simple eigenproblems (with very low flexibility) that are used to
test and develop code"""

from dolfin.cpp.la import PETScMatrix
from dolfin.fem.assembling import assemble
from dolfin.function.functionspace import FunctionSpace

from fenics_utils.mesh import create_unit_hypercube
from fenics_utils.mesh import create_hypercube
from fenics_utils.bcs import set_dirichlet_bcs_lims_all
from fenics_utils.bcs import set_dirichlet_bcs_lims
from fenics_utils.formulation.eigen import formulate_laplacian


def set_free_unit_hypercube(n, diag_value=1e6):
    mesh = create_unit_hypercube(n)
    V = FunctionSpace(mesh, 'P', 1)

    A, B = set_generic(V, bcs=None, diag_value=diag_value)

    return V, A, B


def set_fixed_unit_hypercube(n, diag_value=1e6):
    mesh = create_unit_hypercube(n)
    V = FunctionSpace(mesh, 'P', 1)

    bcs = set_dirichlet_bcs_lims_all(V)
    A, B = set_generic(V, bcs=bcs, diag_value=diag_value)

    return V, A, B


def set_eloi_case(N, dims=[0.2, 0.1, 1.0], axis=2, diag_value=1e6):
    """Box mesh fixed in zlims (or other if axis is passed).

    References:
        [1]: https://cerfacs.fr/coop/fenics-helmholtz
    """
    n = [int(N * dim) for dim in dims]
    mesh = create_hypercube(n, dims)

    V = FunctionSpace(mesh, 'P', 1)

    bcs = set_dirichlet_bcs_lims(V, axis=axis)
    A, B = set_generic(V, bcs=bcs, diag_value=diag_value)

    return V, A, B


def set_generic(V, bcs=None, diag_value=1e6):
    """
    Args:
        bcs_fnc (callable): first argument must be V.
        kwargs: passed to bcs_fnc.
    """

    a, b = formulate_laplacian(V)

    # assemble matrices
    A = PETScMatrix(V.mesh().mpi_comm())
    assemble(a, tensor=A)

    B = PETScMatrix(V.mesh().mpi_comm())
    assemble(b, tensor=B)

    if bcs is None:
        return A, B

    # collect bcs dofs
    bc_dofs = []
    for bc in bcs:
        bc_dofs.extend(list(bc.get_boundary_values().keys()))

    # apply bcs
    A.mat().zeroRowsColumnsLocal(bc_dofs, diag=diag_value)
    # TODO: need to apply in B?
    B.mat().zeroRowsColumnsLocal(bc_dofs, diag=1.)

    return A, B
