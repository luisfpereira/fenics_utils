
import h5py

from dolfin.cpp.mesh import Mesh
from dolfin.cpp.io import XDMFFile
from dolfin.cpp.mesh import MeshFunctionSizet

from dolfin.mesh.meshvaluecollection import MeshValueCollection

import yamio


def get_dolfin_mesh_from_hip(mesh_filename, dolfin_filename):

    # convert mesh
    convert_hip_to_dolfin_mesh(mesh_filename, dolfin_filename)

    # load meshes
    return load_dolfin_mesh(dolfin_filename)


def load_dolfin_mesh(dolfin_filename, with_markers=True):
    mesh = Mesh()
    with XDMFFile(dolfin_filename) as file:
        file.read(mesh)

    if not with_markers:
        return mesh

    # assumes bnd_file exists
    base_filename = '.'.join(dolfin_filename.split('.')[:-1])
    bnd_filename = f'{base_filename}_bnd.xdmf'

    mvc = MeshValueCollection("size_t", mesh)  # subdomains
    with XDMFFile(bnd_filename) as file:
        file.read(mvc, "bnd_patches")

    boundary_markers = MeshFunctionSizet(mesh, mvc)  # boundary parts

    bnd_filename_h5 = f'{base_filename}_bnd.h5'
    with h5py.File(bnd_filename_h5, 'r') as h5_file:
        patch_labels = [name.decode('utf-8').strip() for name in h5_file['PatchLabels'][()]]

    return mesh, boundary_markers, patch_labels


def convert_hip_to_dolfin_mesh(hip_filename, dolfin_filename):
    # expects dolfin to be xdmf
    mesh = yamio.read(hip_filename)
    mesh.write(dolfin_filename, file_format='dolfin-yamio')

    return mesh
