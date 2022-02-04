"""Defines simple eigenproblems (with very low flexibility) that are used to
test and develop code"""

from dolfin.cpp.la import PETScMatrix
from dolfin.fem.assembling import SystemAssembler
from dolfin.fem.assembling import assemble
from dolfin.function.functionspace import FunctionSpace

from fenics_utils.mesh import create_unit_hypercube
from fenics_utils.mesh import create_hypercube
from fenics_utils.bcs import set_dirichlet_bcs_lims_all
from fenics_utils.bcs import set_dirichlet_bcs_lims
from fenics_utils.formulation.eigen import formulate_laplacian


def set_free_unit_hypercube(n):
    mesh = create_unit_hypercube(n)
    V = FunctionSpace(mesh, 'P', 1)
    A, B = set_generic(V, bcs_fnc=None, value=0)

    return V, A, B


def set_fixed_unit_hypercube(n, diag_value=1e6):
    mesh = create_unit_hypercube(n)
    V = FunctionSpace(mesh, 'P', 1)
    A, B = set_generic(V, diag_value, set_dirichlet_bcs_lims_all, value=0.)

    return V, A, B


def set_eloi_case(N, dims=[0.2, 0.1, 1.0], axis=2):
    """Box mesh fixed in zlims (or other if axis is passed).

    References:
        [1]: https://cerfacs.fr/coop/fenics-helmholtz
    """
    n = [int(N * dim) for dim in dims]
    mesh = create_hypercube(n, dims)

    V = FunctionSpace(mesh, 'P', 1)

    A, B = set_generic(V, bcs_fnc=set_dirichlet_bcs_lims, axis=axis)

    return V, A, B


def set_generic(V, diag_value=1e6, bcs_fnc=None, **kwargs):
    """
    Args:
        bcs_fnc (callable): first argument must be V.
        kwargs: passed to bcs_fnc.
    """

    # TODO: avoid use of dummy?
    a, b, dummy = formulate_laplacian(V)

    # bcs
    bcs = bcs_fnc(V, **kwargs) if callable(bcs_fnc) else None

    # assemble
    A, B = _assemble_eigen(V, a, b, dummy, bcs, diag_value)

    return A, B


def _assemble_eigen(V, a, b, dummy, bcs=None, diag_value=1e6):
    if bcs is None:
        A = _assemble_system(V, a, dummy)
    else:
        A = _assemble_system_control_diag(V, a, bcs, diag_value)
    B = _assemble_system(V, b, dummy, bcs)

    return A, B


def _assemble_system(V, a, L, bcs=None):

    asm = SystemAssembler(a, L, bcs)
    A = PETScMatrix(V.mesh().mpi_comm())
    asm.assemble(A)

    return A


def _assemble_system_control_diag(V, a, bcs, diag_value):
    A = PETScMatrix(V.mesh().mpi_comm())
    assemble(a, tensor=A)

    # get bc_dofs
    bc_dofs = []
    for bc in bcs:
        bc_dofs.extend(list(bc.get_boundary_values().keys()))

    # apply bc
    A.mat().zeroRowsColumnsLocal(bc_dofs, diag=diag_value)

    return A
