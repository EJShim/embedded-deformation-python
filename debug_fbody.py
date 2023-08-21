import vtk
from pathlib import Path

def read_polydata(filepath:Path) -> vtk.vtkPolyData:
    reader = vtk.vtkOBJReader()
    reader.SetFileName(filepath)
    reader.Update()

    return reader.GetOutput()

def make_actor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

if __name__ == "__main__":
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)

    polydata = read_polydata("resources/female_body_template.obj")
    print(polydata.GetNumberOfPoints())
    print(polydata.GetNumberOfPolys())
    actor = make_actor(polydata)

    ren.AddActor(actor)
    
    ren.ResetCamera()
    renWin.Render()
    iren.Start()