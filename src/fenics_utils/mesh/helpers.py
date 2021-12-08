
from dolfin.cpp.generation import UnitIntervalMesh
from dolfin.cpp.generation import UnitSquareMesh
from dolfin.cpp.generation import UnitCubeMesh
from dolfin.cpp.generation import IntervalMesh
from dolfin.cpp.generation import RectangleMesh
from dolfin.cpp.generation import BoxMesh

from dolfin.cpp.geometry import Point

from mshr.cpp import Rectangle
from mshr.cpp import Circle
from mshr import generate_mesh


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


def create_rectangle_with_cylinder(point1, point2, cyl_center, cyl_radius,
                                   n_cells=64):
    channel = Rectangle(Point(point1), Point(point2))
    cylinder = Circle(Point(cyl_center), cyl_radius)
    domain = channel - cylinder

    return generate_mesh(domain, n_cells)
