from abc import ABCMeta
from abc import abstractmethod
from lxml import etree

from dolfin.cpp.io import File
from dolfin.cpp.io import XDMFFile
from dolfin.function.function import Function


def convert_xdmf_checkpoints_to_vtk(xdmf_filename, V, var_name,
                                    vtk_filename=None):

    # get times
    times = get_times_from_xdfm_checkpoints(xdmf_filename)

    # create vtk file
    if vtk_filename is None:
        vtk_filename = xdmf_filename.split('.')[0] + '.pvd'
    vtkfile = File(vtk_filename)

    # load variable and save as vtk
    with XDMFFile(xdmf_filename) as file:
        for i, t in enumerate(times):
            u = Function(V, name=var_name)

            file.read_checkpoint(u, var_name, i)

            vtkfile << (u, t)


def load_last_checkpoint(xdmf_filename, V, var_name):

    # get times
    times = get_times_from_xdfm_checkpoints(xdmf_filename)

    # load last checkpoint
    with XDMFFile(xdmf_filename) as file:
        u = Function(V, name=var_name)
        file.read_checkpoint(u, var_name, -1)

    return u, times[-1], len(times) - 1


def get_times_from_xdfm_checkpoints(xdmf_filename):
    # TODO: pass file or filename?
    with open(xdmf_filename, 'r') as file:
        tree = etree.parse(file)

    time_objs = tree.findall('.//Time')

    return [float(time_obj.get('Value')) for time_obj in time_objs]


class OutputsWriter(metaclass=ABCMeta):
    # TODO: extend to non-time dependent cases (although less critical)
    # TODO: should file be an attribute of this?

    @abstractmethod
    def write(self, u, t):
        pass


class VTKWriter(OutputsWriter):

    def __init__(self, file):
        self.file = file

    def write(self, u, t):
        self.file << (u, t)


class XDMFCheckpointWriter(OutputsWriter):

    def __init__(self, file, append_first=False):
        self.file = file
        self._append = append_first  # after first append becomes True

    def write(self, u, t):
        self.file.write_checkpoint(u, u.name(), t, XDMFFile.Encoding.HDF5,
                                   append=self._append)
        self._append = True
