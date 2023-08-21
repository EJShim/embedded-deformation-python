import vtk
from vtk.util import numpy_support
import igl
import numpy as np

if __name__ == "__main__":

    print("IGL read OFF")

    V, F = igl.read_triangle_mesh('decimated-knight.off')

    V = V.astype(np.float)
    F = F.astype(np.int64)
    # a = np.ones((F.shape[0], 1)) * 3
    # F = np.concatenate((a, F), 1).astype(np.int)
    


    v_array = numpy_support.numpy_to_vtk(V)
    f_array = numpy_support.numpy_to_vtk(F.ravel())

    
    points = vtk.vtkPoints()
    points.SetData(v_array)

    faces = vtk.vtkCellArray()
    faces.SetData(3, f_array)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(faces)


    writer = vtk.vtkOBJWriter()
    writer.SetInputData(polydata)
    writer.SetFileName('decimated-knight.obj')
    writer.Write()

    exit()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    iren = vtk.vtkRenderWindowInteractor()
    renWin = vtk.vtkRenderWindow()
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)

    ren.AddActor(actor)
    ren.ResetCamera()

    renWin.Render()
    iren.Start()
    
    # reader = vtk.vtkPLYReader()
    # reader.SetFileName("sample_faust.ply")

    # writer = vtk.vtkOBJWriter()
    # writer.SetInputConnection(reader.GetOutputPort())
    # writer.SetFileName("sample_faust.obj")
    # writer.Write()