

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
