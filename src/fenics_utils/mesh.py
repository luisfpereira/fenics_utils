import os

from pyhip.commands.readers import read_hdf5_mesh
from pyhip.commands.writers import write_gmsh
from pyhip.commands.operations import hip_exit

import meshio

from dolfin.cpp.mesh import Mesh
from dolfin.cpp.generation import UnitIntervalMesh
from dolfin.cpp.generation import UnitSquareMesh
from dolfin.cpp.generation import UnitCubeMesh


def get_dolfin_mesh_from_hdf5(mesh_filename):

    # convert mesh
    dolfin_filename = convert_hdf5_to_dolfin_mesh(mesh_filename)

    # load dolfin mesh
    return Mesh(dolfin_filename)


def convert_hdf5_to_dolfin_mesh(mesh_filename):

    mesh_dirname = os.path.dirname(mesh_filename)
    filename = os.path.basename(mesh_filename).split('.')[0]

    # create gmsh mesh
    gmsh_filename = os.path.join(mesh_dirname, filename)
    read_hdf5_mesh(mesh_filename)
    write_gmsh(gmsh_filename)
    hip_exit()

    # create dolfin mesh
    dolfin_filename = os.path.join(mesh_dirname,
                                   f'{filename}.xml')
    meshio_mesh = meshio.Mesh.read(f'{gmsh_filename}.mesh.msh')
    meshio_mesh.write(dolfin_filename)

    return dolfin_filename


def create_unit_hypercube(n):
    '''Creates a unit mesh with dimensions given by the number of inputs.

    Args:
        n (array-like of int): number of cells.

    Notes:
        Based on Langtangen's book.
    '''

    mesh_obj = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
    return mesh_obj[len(n) - 1](*n)
