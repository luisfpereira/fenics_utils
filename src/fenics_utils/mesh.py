
from dolfin.cpp.generation import UnitIntervalMesh
from dolfin.cpp.generation import UnitSquareMesh
from dolfin.cpp.generation import UnitCubeMesh
from dolfin.cpp.generation import IntervalMesh
from dolfin.cpp.generation import RectangleMesh
from dolfin.cpp.generation import BoxMesh

from dolfin.cpp.geometry import Point


def create_unit_hypercube(n):
    '''Creates a unit mesh with dimensions given by the number of inputs.

    Args:
        n (array-like of int): number of cells.

    Notes:
        Based on Langtangen's book.
    '''

    mesh_obj = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
    return mesh_obj[len(n) - 1](*n)


def create_hypercube(n, dimensions):
    d = len(n)
    if d == 1:
        mesh = IntervalMesh(n[0], 0., dimensions[0])
    else:
        origin = Point([0.] * d)
        point = Point(dimensions)
        mesh_obj = [RectangleMesh, BoxMesh]
        mesh = mesh_obj[d - 2](origin, point, *n)

    return mesh


def get_mesh_axes_lims(mesh):
    mesh_coords = mesh.coordinates()
    return mesh_coords.min(axis=0), mesh_coords.max(axis=0)


def get_mesh_axis_lims(mesh, axis=0):
    mesh_coords = mesh.coordinates()
    x_min, x_max = mesh_coords[:, axis].min(), mesh_coords[:, axis].max()

    return x_min, x_max


def get_mesh_axis_lim(mesh, axis=0, right=True):
    mesh_coords = mesh.coordinates()
    return mesh_coords[:, axis].max() if right else mesh_coords[:, axis].min()
