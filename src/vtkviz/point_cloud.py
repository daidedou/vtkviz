import numpy as np
import vtk
from .utils import VTKEntity3D


class VTKPointCloud(VTKEntity3D):
    def __init__(self, points: np.ndarray=None, colors: np.ndarray=None):
        assert (points is None and colors is None) or (points is not None and colors is not None)

        self.num_points = 0

        # VTK geometry representation
        self._points = vtk.vtkPoints()

        # VTK color representation
        self._colors = vtk.vtkUnsignedCharArray()
        self._colors.SetName("Colors")
        self._colors.SetNumberOfComponents(3)

        # Visualization pipeline
        # - Data source
        point_data = vtk.vtkPolyData()
        point_data.SetPoints(self._points)
        point_data.GetPointData().SetScalars(self._colors)

        # - Automatically generate topology cells from points
        mask_points = vtk.vtkMaskPoints()
        mask_points.SetInputData(point_data)
        mask_points.GenerateVerticesOn()
        mask_points.SingleVertexPerCellOn()
        mask_points.Update()

        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mask_points.GetOutputPort())

        super().__init__(mapper)

        self.add_points(points, colors)

    def add_points(self, points: np.ndarray, colors: np.ndarray): #points, colors Nx3
        assert len(points.shape) == 2
        assert len(colors.shape) == 2
        assert points.shape[0] == colors.shape[0]
        assert points.shape[1] == 3
        assert colors.shape[1] == 3

        [num_new_points, _] = points.shape

        # Allocate additional memory
        self._points.Resize(self.num_points + num_new_points * 2)
        self._colors.Resize(self.num_points + num_new_points * 2)

        # Add points
        for point_idx in range(num_new_points):
            for _ in range(2):  # add every point twice due to vtk bug
                self._points.InsertNextPoint(points[point_idx, :])
                self._colors.InsertNextTuple(colors[point_idx, :])

        self._points.Modified()
        self._colors.Modified()

    def set_point_size(self, size: int):
        self.actor.GetProperty().SetPointSize(size)


class VTKPointCloudSphere(VTKEntity3D):
    def __init__(self, points: np.ndarray=None, colors: np.ndarray=None, scale=1):
        assert (points is None and colors is None) or (points is not None and colors is not None)

        self.num_points = 0

        # VTK geometry representation
        self._points = vtk.vtkPoints()

        # VTK color representation
        self._colors = vtk.vtkUnsignedCharArray()
        self._colors.SetName("Colors")
        self._colors.SetNumberOfComponents(3)
        # Create scale factors (single value for all points)
        self.scale_factors = vtk.vtkFloatArray()
        self.scale_factors.SetNumberOfComponents(1)
        self.scale_factors.SetName("Scale Factor")
        self.scale = scale

        # Visualization pipeline
        # - Data source
        point_data = vtk.vtkPolyData()
        point_data.SetPoints(self._points)
        point_data.GetPointData().AddArray(self._colors)
        point_data.GetPointData().AddArray(self.scale_factors)

        sphere_source = vtk.vtkSphereSource()
        # sphere_source.SetThetaResolution(20)
        # sphere_source.SetPhiResolution(20)
        # - Automatically generate topology cells from points
        glyph3D_mapper = vtk.vtkGlyph3DMapper()
        glyph3D_mapper.SetSourceConnection(sphere_source.GetOutputPort())
        glyph3D_mapper.SetInputData(point_data)
        glyph3D_mapper.SetScaleModeToScaleByMagnitude()
        glyph3D_mapper.SetScaleArray("Scale Factor")
        glyph3D_mapper.SetScalarModeToUsePointFieldData()
        glyph3D_mapper.SelectColorArray("Colors")
        glyph3D_mapper.Update()

        # - Map the data representation to graphics primitives

        super().__init__(glyph3D_mapper)

        self.add_points(points, colors)

    def add_points(self, points: np.ndarray, colors: np.ndarray): #points, colors Nx3
        assert len(points.shape) == 2
        assert len(colors.shape) == 2
        assert points.shape[0] == colors.shape[0]
        assert points.shape[1] == 3
        assert colors.shape[1] == 3

        [num_new_points, _] = points.shape

        # Allocate additional memory
        self._points.Resize(self.num_points + num_new_points * 2)
        self._colors.Resize(self.num_points + num_new_points * 2)

        # Add points
        for point_idx in range(num_new_points):
            for _ in range(2):  # add every point twice due to vtk bug
                self._points.InsertNextPoint(points[point_idx, :])
                self._colors.InsertNextTuple(colors[point_idx, :])
                self.scale_factors.InsertNextValue(self.scale)

        self._points.Modified()
        self._colors.Modified()

    def set_point_size(self, size: int):
        self.actor.GetProperty().SetPointSize(size)


class VTKVectorField(VTKEntity3D):
    def __init__(self, positions: np.ndarray, vectors: np.ndarray, color=[1, 0, 0]):
        self.num_vectors = 0

        # VTK position representation
        self._positions = vtk.vtkPoints()

        # VTK vector representation
        self._vectors = vtk.vtkFloatArray()
        self._vectors.SetName("Vector Field")
        self._vectors.SetNumberOfComponents(3)

        # VTK norm vector representation
        self._vectors_norm = vtk.vtkFloatArray()
        self._vectors_norm.SetName("Vector Field")


        # Visualization Pipeline
        # - Data source
        position_data = vtk.vtkPolyData()
        position_data.SetPoints(self._positions)
        position_data.GetPointData().AddArray(self._vectors)
        position_data.GetPointData().SetActiveVectors("Vector Field")

        # - Add the vector arrays as 3D Glyphs

        grid = vtk.vtkPolyData()
        grid.SetPoints(self._positions)
        grid.GetPointData().SetVectors(self._vectors)

        #grid.GetPointData().SetScalars(self._vectors_norm)

        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(16)
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.1)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(grid)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        #glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetColorModeToColorByScalar()
        glyph.OrientOn()

        glyph.Update()

        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        #mapper.SetInputConnection(add_arrows.GetOutputPort())
        mapper.SetInputConnection(glyph.GetOutputPort())

        super().__init__(mapper)
        if color is not None:
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])

        self.add_vectors(positions, vectors)

    def add_vectors(self, positions: np.ndarray, vectors: np.ndarray):
        assert len(positions.shape) == 2
        assert len(vectors.shape) == 2
        assert positions.shape[0] == vectors.shape[0]
        assert positions.shape[1] == 3
        assert vectors.shape[1] == 3

        [num_new_vectors, _] = vectors.shape

        # Allocate additional memory
        self._positions.Resize(self.num_vectors + num_new_vectors)
        self._vectors.Resize(self.num_vectors + num_new_vectors)
        self._vectors_norm.Resize(self.num_vectors + num_new_vectors)

        # Add points
        for vector_idx in range(num_new_vectors):
            self._positions.InsertNextPoint(positions[vector_idx, :])
            self._vectors.InsertNextTuple(vectors[vector_idx, :])
            self._vectors_norm.InsertNextValue(np.linalg.norm(vectors[vector_idx, :]))

        self._positions.Modified()
        self._vectors.Modified()
        self._vectors_norm.Modified()

