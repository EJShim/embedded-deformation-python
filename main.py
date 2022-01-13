import vtk
from vtk.util import numpy_support
import numpy as np


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


pointBuffer = np.array([
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0],
        [5, 0, 0],        
    ], dtype=np.float64)
polydata = vtk.vtkPolyData()
cubeGlyph = []

def MakeSphereActor(polydata):
    mapper = vtk.vtkOpenGLSphereMapper()
    mapper.SetRadius(0.1)
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def MakeActor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


pickedId = -1

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
    pointBuffer[pickedId][0] = pos[0]
    pointBuffer[pickedId][1] = pos[1]
    pointBuffer[pickedId][2] = pos[2]
    polydata.GetPoints().Modified()

    #Update Glyph
    cubeGlyph[pickedId].SetPosition(pos)

    renWin.Render()

def MakeBoxGlyph():
    source = vtk.vtkCubeSource()
    source.SetCenter(0, 0, 0)
    source.SetXLength(.6)
    source.SetYLength(.6)
    source.SetZLength(.6)
    source.Update()

    actor = MakeActor(source.GetOutput())
    actor.GetProperty().SetColor(1, 0, 0)
    actor.GetProperty().SetOpacity(.2)

    return actor
    

    

def MouseReleaseCallback(obj, e):    
    global pickedId    
    pickedId = -1    

if __name__ == "__main__":    

    points = vtk.vtkPoints()
    pointArray = numpy_support.numpy_to_vtk(pointBuffer)
    points.SetData(pointArray)

    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(pointBuffer.shape[0])
    for pid in range(pointBuffer.shape[0]):
        line.GetPointIds().SetId(pid, pid)
        cube = MakeBoxGlyph()
        cube.SetPosition(pointBuffer[pid])
        cubeGlyph.append(cube)
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(line)
    #vtkPolyData -> 3차원 오브젝트 정보를 가지고있는 객체    
    polydata.SetPoints(points)
    polydata.SetLines(polys)


    pointActor = MakeSphereActor(polydata)
    lineActor = MakeActor(polydata)

    #렌더러에 오브젝트 추가
    ren.AddActor(pointActor)
    ren.AddActor(lineActor)
    for cube in cubeGlyph:
        ren.AddActor(cube)


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
