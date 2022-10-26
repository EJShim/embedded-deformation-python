from distutils.archive_util import make_archive
import vtk
import argparse
from vtk.util import numpy_support
import numpy as np
import utils
import igl



    

def make_polydata(V, F) -> vtk.vtkPolyData:
    V = np.array(V, dtype=np.float32).copy()
    F = np.array(F, dtype=np.int32).copy()


    points = vtk.vtkPoints()
    points_data = numpy_support.numpy_to_vtk(V)
    points.SetData(points_data)

    triangles = vtk.vtkCellArray()
    # cell_data = numpy_support.numpy_to_vtk(F.ravel()) ## not working.... outside of this function
    # triangles.SetData(3, cell_data)
    for face in F:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, face[0])
        triangle.GetPointIds().SetId(1, face[1])
        triangle.GetPointIds().SetId(2, face[2])
        triangles.InsertNextCell(triangle)


    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    return polydata

def read_polydata(path: str) ->vtk.vtkPolyData:
    
    ext = path.split(".")[-1]

    if ext == "off":
        # Read Data
        V, F = igl.read_triangle_mesh(path)

        # Make vtkpolydata
        poly = make_polydata(V, F)

        
        return poly
    else:
        raise("only .off file is supoprted for noaw")




if __name__ == "__main__":    
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='resources/decimated-knight.off')
    args = parser.parse_args()

    #기본 셋업
    iren = vtk.vtkRenderWindowInteractor()
    interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(interactorStyle)
    renWin = vtk.vtkRenderWindow()
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)
    ren.SetBackground(0.1, 0.2, 0.4)
    renWin.SetSize(1000, 1000)

    polydata = read_polydata(args.input)

    actor = utils.MakeActor(polydata=polydata)
    
    ren.AddActor(actor)
    ren.ResetCamera()



    #refresh
    ren.ResetCamera()
    renWin.Render()

    #렌더러 창 실행
    iren.Start()
