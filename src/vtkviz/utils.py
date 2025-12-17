from abc import ABC
import vtk
import numpy as np

# This is just that y and z may be inverted depending on what you are used to.
dir_x = np.array([1, 0, 0])
dir_y = np.array([0, 0, 1])
dir_z = np.array([0, 1, 0])

def get_p1p2(vertices, given_dir):
    prod = np.inner(vertices, given_dir)
    p_max = np.amax(prod)
    p_min = np.amin(prod)
    p1 = p_max + 0.1 * (p_max - p_min)
    p2 = p_min + 0.1 * (p_min - p_max)
    return p1, p2

def center_mesh_give_size(vertices, two_dirs=False):
    p1_z, p2_z = get_p1p2(vertices, dir_z)
    if two_dirs:
        p1_x, p2_x = get_p1p2(vertices, dir_x)
        if abs(p1_x - p2_x) > abs(p1_z - p2_z):
            return p1_x, p2_x
        else:
            return p1_z, p2_z
    return p1_z, p2_z



def ReadImage(file_name):
    if ".png" in file_name:
        reader = vtk.vtkPNGReader()
    else:
        reader = vtk.vtkJPEGReader()
    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()

def ReadPolyData(file_name):
    import os
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()
    if extension == ".ply":
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLpoly_dataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".g":
        reader = vtk.vtkBYUReader()
        reader.SetGeometryFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    else:
        # Return a None if the extension is unknown.
        poly_data = None
    return poly_data

class VTKEntity3D(ABC):
    def __init__(self, mapper: vtk.vtkMapper):
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(mapper)

    @property
    def actor(self):
        return self._actor