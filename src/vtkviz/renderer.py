import vtk
import numpy as np
from .utils import VTKEntity3D

from vtkmodules.vtkCommonColor import vtkNamedColors

colors = vtkNamedColors()
colors.SetColor('HighNoonSun', [255, 255, 251, 255])
colors.SetColor('100W Tungsten', [255, 214, 170, 255])



class VTKVisualization(object):
    def __init__(self, set_axes=False):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(colors.GetColor3d('Silver'))#1., 1., .9)
        self.renderer.ResetCamera()

        if set_axes:
            axes_actor = vtk.vtkAxesActor()
            axes_actor.AxisLabelsOff()
            self.renderer.AddActor(axes_actor)

        self.window = None

        self.camera = vtk.vtkCamera()
        self.camera.SetViewUp(0.0, 1.0, 0.0)
        self.camera.SetPosition(0.0, 0.0, +5)
        self.camera.SetFocalPoint(0.0, 0.0, 0.0)
        # self.camera.SetClippingRange(0.0, 100000)"""

        self.renderer.SetActiveCamera(self.camera)

    def add_entity(self, entity: VTKEntity3D):
        self.renderer.AddActor(entity.actor)

    def add_image(self, image):
        img_mapper = vtk.vtkImageMapper()
        img_actor = vtk.vtkActor2D()
        img_data = vtk.vtkImageData()
        img_data.SetDimensions(image.shape[0], image.shape[1], 1)
        img_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                pixel = img_data.GetScalarPointer(x, y, 0)
                pixel = np.array(image[x, y, :])
        img_mapper.SetInputData(img_data)
        img_mapper.SetColorWindow(255)
        img_mapper.SetColorLevel(127.5)
        img_actor.SetMapper(img_mapper)
        self.renderer.AddActor(img_actor)

    def init(self, interactor=None, size=None):
        if interactor is None:
            self.window = vtk.vtkRenderWindow()
            self.window.AddRenderer(self.renderer)
            self.window.Render()
            if size is None:
                self.window.SetSize(1200, 800)
            else:
                self.window.SetSize(size[0], size[1])
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.interactor.SetRenderWindow(self.window)
            self.iren = self.window.GetInteractor()
        else:
            self.interactor = interactor
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.window = interactor.GetRenderWindow()
            self.window.AddRenderer(self.renderer)
            self.iren = self.window.GetInteractor()

    def show(self):

        self.interactor.Start()
        self.iren.Initialize()

    def write(self, filename, background=None, size=None):
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetAlphaBitPlanes(1)
        if size is None:
            self.window.SetSize(1800, 1800)
        else:
            self.window.SetSize(size[0], size[1])
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        if background is None:
            windowToImageFilter.SetInputBufferTypeToRGBA()
        else:
            self.renderer.SetBackground(background[0], background[1], background[2])
        self.window.AddRenderer(self.renderer)
        windowToImageFilter.SetInput(self.window)
        windowToImageFilter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()

    def reset_camera(self):
        self.renderer.ResetCamera()

    def change_camera(self, view, position, focal):
        self.camera = vtk.vtkCamera()
        self.camera.SetViewUp(view[0], view[1], view[2])
        self.camera.SetPosition(position[0], position[1], position[2])
        self.camera.SetFocalPoint(focal[0], focal[0], focal[0])
        self.camera.Pitch(0)
        self.camera.OrthogonalizeViewUp()
        # self.camera.SetClippingRange(0.0, 100000)

        self.renderer.SetActiveCamera(self.camera)


class VTKMultipleVizualization(object):
    # Not necessarily useful for everybody
    # A class to vizualize multiple object in the same window
    # Most ideas come from https://github.com/SRNFmatch/SRNFmatch_code/blob/master/src/input_output.py,
    # PlotGeodesic (and other vizualization functions)
    def __init__(self, N, sync=True):
        q, r = np.divmod(N, 4)

        if N <= 4:
            L_ind = np.arange(N)
            n_hwin = N
            n_vwin = 1
        elif N > 4 and N <= 8:
            L_ind = np.arange(N)
            n_hwin = 4
            n_vwin = 2
        else:
            L_ind = np.linspace(0, N - 1, 8)
            L_ind = np.rint(L_ind)
            L_ind = L_ind.astype(int)
            Nt = 8
            n_hwin = 4
            n_vwin = 2
        self.L_ind = L_ind
        self.window = vtk.vtkRenderWindow()

        self.camera = vtk.vtkCamera()
        self.camera.SetViewUp(0.0, 1.0, 0.0)
        self.camera.SetPosition(0.0, 0.0, +2.5)
        self.camera.SetFocalPoint(0.0, 0.0, 0.0)
        # self.camera.SetClippingRange(0.0, 10000)"""
        self.renderers = []
        for i in range(N):
            qi, ri = np.divmod(i, 4)
            ren = vtk.vtkRenderer()
            ren.SetBackground(0.5, 0.5, 0.5)
            self.window.AddRenderer(ren)
            ren.SetViewport(ri / n_hwin, 1 - (qi + 1) / n_vwin, (ri + 1) / n_hwin, 1 - qi / n_vwin)
            self.renderers.append(ren)
        if sync:
            for i in range(N - 1):
                self.renderers[i].SetActiveCamera(self.renderers[-1].GetActiveCamera())

    def show(self):
        self.window.Render()
        #self.window.SetSize(1200, 800)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.SetRenderWindow(self.window)
        self.interactor.Start()

    def add_entities(self, entities):
        for i, entity in enumerate(entities):
            self.renderers[i].AddActor(entity.actor)

    def add_entity(self, entity, i):
        self.renderers[i].AddActor(entity.actor)