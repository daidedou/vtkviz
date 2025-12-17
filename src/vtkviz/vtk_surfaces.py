from .utils import VTKEntity3D, ReadPolyData, ReadImage
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa


def poly_to_numpy(polydata):
    # https://stackoverflow.com/a/50507141
    vertices = dsa.WrapDataObject(polydata).Points
    # We could actually use dsa.WrapDataObject(polydata).Polygons.reshape((-1, 4))[:, 1:]
    # But I rather check if we have only triangles as vtk allows for more versatile structure
    # https://stackoverflow.com/a/51371909
    cells = polydata.GetPolys()
    nCells = cells.GetNumberOfCells()
    array = cells.GetData()
    # This holds true if all polys are of the same kind, e.g. triangles.
    assert (array.GetNumberOfValues() % nCells == 0)
    nCols = array.GetNumberOfValues() // nCells
    numpy_cells = vtk_to_numpy(array)
    faces = numpy_cells.reshape((-1, nCols))[:, 1:]
    return vertices, faces


def decimate(surf_actor, verbose=False, reduction_factor=0.1):
    decimate = vtk.vtkQuadricDecimation()
    assert reduction_factor > 0 and reduction_factor <= 1, "reduction_factor should be between 0 and 1"
    if verbose:
        print("Before decimation \n"
              "-----------------\n"
              "There are " + str(surf_actor.surface_data.GetNumberOfPoints()) + "points.\n"
                                                                 "There are " + str(
            surf_actor.surface_data.GetNumberOfPolys()) + "polygons.\n")
        print("Reduction Factor: " + str(reduction_factor))
    decimate.SetInputData(surf_actor.surface_data)
    decimate.SetTargetReduction(1-reduction_factor)
    decimate.Update()
    polydata = decimate.GetOutput()
    poly_actor = VTKPoly(polydata)
    if verbose:
        print("After decimation \n"
              "-----------------\n"
              "There are " + str(polydata.GetNumberOfPoints()) + "points.\n"
                                                                      "There are " + str(
            polydata.GetNumberOfPolys()) + "polygons.\n")
    # https://stackoverflow.com/a/50507141
    red_vertices, red_faces = poly_to_numpy(polydata)
    return poly_actor, red_vertices, red_faces


def gen_tex_coords(vertices, axes=(0, 1)):
    """
    Ref:
    - https://github.com/llorz/SGA19_zoomOut/blob/master/utils/%2BMESH/%2BMESH_IO/generate_tex_coords.m

    Args:
        vertices (np.ndarray): (V, 3)
        axes (tuple): plane of axes[0] and axes[1]

    Returns:
        np.ndarray: (V, 2)
    """
    vt = vertices[:, axes]  # (V, 2)
    vt = vt - np.amin(vt, axis=0, keepdims=True)
    vt = vt / np.amax(vt)
    # Offset
    vt = ((vt * 2 - 1) * 0.9 + 1) / 2
    return vt


class VTKPoly(VTKEntity3D):
    def __init__(self, poly):
        self.polydata = poly
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        super().__init__(mapper)

    def update_poly(self, poly):
        self.polydata.ShallowCopy(poly)
        self.polydata.Modified()



class VTKFile(VTKEntity3D):
    def __init__(self, filename, shade=False, angle_shade=0):
        polydata = ReadPolyData(filename)
        if shade:
            normalGenerator = vtk.vtkPolyDataNormals()
            normalGenerator.SetInputData(polydata)
            normalGenerator.ComputePointNormalsOn()
            normalGenerator.SetFeatureAngle(angle_shade)
            normalGenerator.Update()
            normalGenerator.SplittingOn()

            normalsPolyData = vtk.vtkPolyData()
            normalsPolyData.DeepCopy(normalGenerator.GetOutput())
            self.polydata = normalsPolyData
        else:
            self.polydata = polydata
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.polydata)

        super().__init__(mapper)

    def update_file(self, filename):
        new_poly = ReadPolyData(filename)
        self.polydata.ShallowCopy(new_poly)
        self.polydata.Modified()

    def update_poly(self, poly):
        self.polydata.ShallowCopy(poly)
        self.polydata.Modified()


class VTKSurface(VTKEntity3D):
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, color=None, uv=None, shade=False, angle_shade=0.):
        self.num_vertices = 0
        self.num_faces = 0

        # VTK point cloud representation
        self._vertices = vtk.vtkPoints()

        # VTK polygone(surface) representation
        self._faces = vtk.vtkCellArray()

        # Visualization Pipeline
        # - Data source
        self.surface_data = vtk.vtkPolyData()
        self.surface_data.SetPoints(self._vertices)
        self.surface_data.SetPolys(self._faces)

        # - Add the vector arrays as 3D Glyphs
        self.normalGenerator = vtk.vtkPolyDataNormals()
        self.normalGenerator.SetInputData(self.surface_data)
        self.normalGenerator.ComputePointNormalsOn()
        self.normalGenerator.SetFeatureAngle(angle_shade)
        self.normalGenerator.Update()
        self.normalGenerator.SplittingOn()
        self.normals_data = self.normalGenerator.GetOutput()

        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.normals_data)
        mapper.ScalarVisibilityOff()

        super().__init__(mapper)
        if color is None:
            super().__init__(mapper)
            self.actor.GetProperty().SetColor(0.5, 0.5, 1.0)
            self.actor.GetProperty().SetOpacity(1)
        elif type(color)== list:
            super().__init__(mapper)
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])
            if len(color) == 4:
                print("Setting opacity: ", color[3])
                self.actor.GetProperty().SetOpacity(color[3])
        elif type(color)==str:
            if uv is None:
                uv = gen_tex_coords(vertices)
            self.surface_data.GetPointData().SetTCoords(numpy_to_vtk(uv))
            img_data = ReadImage(color)
            tu = vtk.vtkTexture()
            tu.SetInputData(img_data)
            tu.SetInterpolate(False)
            self.actor.SetTexture(tu)
        elif len(color.shape) == 1:
            colors_array = vtk.vtkDoubleArray()
            for i in range(color.shape[0]):
                colors_array.InsertNextValue(color[i])
            self.surface_data.GetPointData().SetScalars(colors_array)
            self.lut = vtk.vtkLookupTable()
            range_data = self.surface_data.GetPointData().GetScalars().GetRange()
            self.lut.SetTableRange(range_data)
            self.lut.SetHueRange(0.1, 0.35)
            self.lut.SetSaturationRange(.8, 1.)
            self.lut.SetValueRange(0.7, 1.0)
            mapper.SetLookupTable(self.lut)
            mapper.SetScalarRange(range_data)
            mapper.Update()
            super().__init__(mapper)
            self.actor.GetProperty().SetOpacity(1)
        else:
            colors_array = vtk.vtkUnsignedCharArray()
            colors_array.SetNumberOfComponents(3)
            for i in range(color.shape[0]):
                colors_array.InsertTuple(i, color[i, :])
            self.surface_data.GetPointData().SetScalars(colors_array)
            mapper.ScalarVisibilityOn()
            #super.__init__(mapper)
            self.actor.GetProperty().SetOpacity(1)

        if shade:
            self.actor.GetProperty().SetInterpolationToGouraud()
        else:
            self.actor.GetProperty().SetInterpolationToFlat()

        self.add_vectors(vertices, faces)

    def add_vectors(self, vertices: np.ndarray, faces: np.ndarray):
        assert len(vertices.shape) == 2
        assert len(faces.shape) == 2
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

        # Add points
        points = np.vstack(vertices)
        vtkArray = numpy_to_vtk(np.ascontiguousarray(points), deep=True)  # , deep=True)
        self._vertices.SetData(vtkArray)
        [num_faces, _] = faces.shape
        insert_points = range(0, num_faces * 3, 3)
        faces_vtk_format = np.insert(faces.ravel(), insert_points, 3).astype(np.int64)
        vtkfacesArray = numpy_to_vtkIdTypeArray(faces_vtk_format, deep=True)  # , deep=True)
        self._faces.SetCells(faces.shape[0], vtkfacesArray)
        self.surface_data.SetPolys(self._faces)
        self._vertices.Modified()

        self._faces.Modified()
        self.surface_data.SetPolys(self._faces)
        self.normalGenerator.Update()

    def change_points(self, vertices: np.ndarray, faces: np.ndarray):
        assert len(vertices.shape) == 2
        assert len(faces.shape) == 2
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

        # Add points
        points = np.vstack(vertices)
        vtkArray = numpy_to_vtk(np.ascontiguousarray(points), deep=True)  # , deep=True)
        self._vertices.SetData(vtkArray)

        insert_points = range(0, len(faces) * 3, 3)
        faces_vtk_format = np.insert(faces.ravel(), insert_points, 3).astype(np.int64)
        vtkfacesArray = numpy_to_vtkIdTypeArray(faces_vtk_format, deep=True)  # , deep=True)
        self._faces.SetCells(faces.shape[0], vtkfacesArray)
        self.surface_data.SetPolys(self._faces)
        self._vertices.Modified()

        self._faces.Modified()
        self.surface_data.SetPolys(self._faces)
        self.normalGenerator.Update()

    def updateVertices(self, vertices):
        assert vertices.shape[1] == 3
        points = np.vstack(vertices)
        vtkArray = numpy_to_vtk(np.ascontiguousarray(points), deep=True)  # , deep=True)
        self._vertices.SetData(vtkArray)
        self._vertices.Modified()
        self.normalGenerator.Update()


    def updateTexture(self, uv):
        self.surface_data.GetPointData().SetTCoords(numpy_to_vtk(uv))