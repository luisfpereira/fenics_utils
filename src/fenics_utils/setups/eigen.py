'''Defines simple eigenproblems (with very low flexibility) that are used to
test and develop code'''

from dolfin.cpp.la import PETScMatrix
from dolfin.fem.assembling import SystemAssembler

from fenics_utils.mesh import create_unit_hypercube
from fenics_utils.formulation.eigen import formulate_laplacian


def set_free_unit_hypercube(n, V=None):

    mesh = create_unit_hypercube(n)

    V, a, b, dummy = formulate_laplacian(mesh, V)
    A, B = _assemble_eigen(a, b, dummy)

    return V, A, B


def _assemble_eigen(a, b, dummy, bcs=None):
    A = _assemble_system(a, dummy, bcs)
    B = _assemble_system(b, dummy, bcs)

    return A, B


def _assemble_system(a, L, bcs=None):
    asm = SystemAssembler(a, L, bcs)
    A = PETScMatrix()
    asm.assemble(A)

    return A
