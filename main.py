import vtk
from vtk.util import numpy_support
import numpy as np
import math
import utils


#기본 셋업
iren = vtk.vtkRenderWindowInteractor()
interactorStyle = vtk.vtkInteractorStyleImage()
iren.SetInteractorStyle(interactorStyle)
renWin = vtk.vtkRenderWindow()
iren.SetRenderWindow(renWin)
ren = vtk.vtkRenderer()
renWin.AddRenderer(ren)
ren.SetBackground(0.1, 0.2, 0.4)
renWin.SetSize(1000, 1000)


pickedId = -1

targetGraph = None

def MouseClickCallback(obj, e):    
    global pickedId
    
    
    picker = obj.GetPicker()
    eventPosition = obj.GetEventPosition()
    picker.Pick(float(eventPosition[0]),float(eventPosition[1]),0.0,ren)    
    pickedId = picker.GetPointId()    

    

def MouseMoveCallback(obj, r):
    
    if pickedId == -1 : return    
    picker = obj.GetPicker()
    eventPosition = obj.GetEventPosition()
    picker.Pick(float(eventPosition[0]),float(eventPosition[1]),0.0,ren)    

    #Update Point
    pos = picker.GetPickPosition()

    targetGraph.updatePoint(pickedId, pos)

    renWin.Render()


    

def MouseReleaseCallback(obj, e):    
    global pickedId    
    pickedId = -1    
    transBuffer = np.zeros_like(pointBuffer)

if __name__ == "__main__":    

    pointBuffer = np.array([
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0],
        [5, 0, 0],        
    ], dtype=np.float64)
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    pointArray = numpy_support.numpy_to_vtk(pointBuffer)
    points.SetData(pointArray)

    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(pointBuffer.shape[0])
    for pid in range(pointBuffer.shape[0]):
        line.GetPointIds().SetId(pid, pid)        
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(line)    
    polydata.SetPoints(points)
    polydata.SetLines(polys)

    targetGraph = utils.DeformableGraph(polydata)
    targetGraph.addToRenderer(ren)



    #refresh
    ren.ResetCamera()
    renWin.Render()

    #Add Interaction    
    picker = vtk.vtkPointPicker()
    picker.SetTolerance(0.01)
    iren.SetPicker(picker)

    iren.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, MouseClickCallback)    
    iren.AddObserver(vtk.vtkCommand.InteractionEvent, MouseMoveCallback)
    iren.AddObserver(vtk.vtkCommand.EndInteractionEvent , MouseReleaseCallback)

    #렌더러 창 실행
    iren.Initialize()
    iren.Start()
