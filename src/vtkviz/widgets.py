import vtk
import numpy as np
from .utils import VTKEntity3D

class VTKSlider:
    def __init__(self, name, min_val=0., max_val=1.0, pos=(100, 700), length=300):
        self.slideBar = vtk.vtkSliderRepresentation2D()

        self.slideBar.SetMinimumValue(min_val)
        self.slideBar.SetMaximumValue(max_val)
        self.slideBar.SetTitleText(name)

        self.slideBar.GetSliderProperty().SetColor(1, 1, 1)  # Couleur endroit select (quand pas cliqué)
        self.slideBar.GetTitleProperty().SetColor(1, 0, 0)  # Couleur titre
        self.slideBar.GetLabelProperty().SetColor(1, 0, 0)  # Couleur valeur select
        self.slideBar.GetSelectedProperty().SetColor(1, 0, 0)  # Couleur endroit slider
        self.slideBar.GetTubeProperty().SetColor(0, 1, 0)  # Slider fond
        self.slideBar.GetCapProperty().SetColor(1, 1, 0)  # Extremités

        self.slideBar.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
        self.slideBar.GetPoint1Coordinate().SetValue(pos[0], pos[1])  # Endroit dans la scène

        self.slideBar.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
        self.slideBar.GetPoint2Coordinate().SetValue(pos[0]+length, pos[1])  # Endroit dans la scène

    def enable_widget(self, interactor):
        print("defining")
        self.sliderWidget = vtk.vtkSliderWidget()
        print("setting interactor")
        self.sliderWidget.SetInteractor(interactor)
        print("representation")
        self.sliderWidget.SetRepresentation(self.slideBar)
        print("enabling")
        self.sliderWidget.EnabledOn()


    def add_callback(self, callback):
        self.sliderWidget.AddObserver("InteractionEvent", callback)


class VTKHull(VTKEntity3D):
    def __init__(self, convex_hull, color=None):
        # - convex hull is the result of delaunay triangulation
        self.convex_hull = convex_hull
        # - Map the data representation to graphics primitives
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(self.convex_hull.GetOutput())

        super().__init__(mapper)
        if color is None:
            self.actor.GetProperty().SetColor(0.5, 0.5, 1.0)
        else:
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])
        self.actor.GetProperty().SetOpacity(0.5)


class VTKPlane(VTKEntity3D):
    def __init__(self, center: np.ndarray, normal: np.ndarray):

        # VTK point cloud representation
        self._plane = vtk.vtkPlaneSource()
        self._plane.SetCenter(center)
        self._plane.SetNormal(normal)
        self._plane.Update()

        # VTK polygone(surface) representation
        self._faces = vtk.vtkCellArray()

        # Visualization Pipeline
        # - Data source
        plane_data = self._plane.GetOutput()
        # - Add the vector arrays as 3D Glyphs


        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(plane_data)

        super().__init__(mapper)

class VTKLine(VTKEntity3D):
    def __init__(self, orig: np.ndarray, end: np.ndarray, color=None):

        # VTK point cloud representation
        self._line = vtk.vtkLineSource()
        self._line.SetPoint1(orig)
        self._line.SetPoint2(end)
        self._line.Update()

        # Visualization Pipeline
        # - Data source
        line_data = self._line.GetOutput()
        # - Add the vector arrays as 3D Glyphs


        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(line_data)

        super().__init__(mapper)
        if color is None:
            self.actor.GetProperty().SetColor(1.0, 0., 0.)
        else:
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])


class VTKTubeLine(VTKEntity3D):
    def __init__(self, orig: np.ndarray, end: np.ndarray, color=None):

        # VTK point cloud representation
        self._line = vtk.vtkLineSource()
        self._line.SetPoint1(orig)
        self._line.SetPoint2(end)
        self._line.Update()

        self.tubeFilter = vtk.vtkTubeFilter()
        self.tubeFilter.SetInputData(self._line.GetOutput())
        self.tubeFilter.SetRadius(.025) # default is .5
        self.tubeFilter.SetNumberOfSides(50)
        self.tubeFilter.Update()
        # Visualization Pipeline
        # - Data source
        tube_data = self.tubeFilter.GetOutput()
        # - Add the vector arrays as 3D Glyphs


        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(tube_data)

        super().__init__(mapper)
        if color is None:
            self.actor.GetProperty().SetColor(1.0, 0., 0.)
        else:
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])


