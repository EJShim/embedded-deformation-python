import vtk
import igl
from vtk.util import numpy_support
import numpy as np

def make_actor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

def build_triangle_mesh(v, f):

    polydata = vtk.vtkPolyData()

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(v))
    polydata.SetPoints(points)

    polys = vtk.vtkCellArray()
    polys.SetData(3,  numpy_support.numpy_to_vtk(f.ravel()))
    polydata.SetPolys(polys)

    return polydata

def extract_surface_index(v, f):
    # current V has vertices of tetrahedra, only use surface v here
    _,_,_, J = igl.remove_unreferenced(v,f)
    
    return J

def build_tet_mesh(v, t):
    tet = vtk.vtkUnstructuredGrid()

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(v))
    tet.SetPoints(points)

    polys = vtk.vtkCellArray()
    polys.SetData(4,  numpy_support.numpy_to_vtk(t.ravel()))
    tet.SetCells(10,polys)

    return tet

def compute_biharmonic(low_v, high_v, high_t):
    J = np.arange(high_v.shape[0])

    # Point Matching
    _, b, _ = igl.point_mesh_squared_distance(low_v, high_v, J)
    S = np.expand_dims(b, 1)
    
    W = igl.biharmonic_coordinates(high_v, high_t, S, 3)

    return W

if __name__ == "__main__":
    iren  = vtk.vtkRenderWindowInteractor()
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()    
    iren.SetInteractorStyle(interactor_style)
    renWin = vtk.vtkRenderWindow()
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)

    # Read low and high mesh
    low_v, low_t, low_f = igl.read_mesh("resources/octopus-low.mesh")
    high_v, high_t, high_f = igl.read_mesh("resources/octopus-high.mesh")

    # Render Low Mesh
    low_J = extract_surface_index(low_v, low_f)
    low_v_s = low_v[low_J]
    low_poly = build_triangle_mesh(low_v_s, low_f)
    low_actor = make_actor(low_poly)
    ren.AddActor(low_actor)

    # Render High Mesh
    high_J = extract_surface_index(high_v, high_f)
    high_v_s = high_v[high_J]
    

    print("Calculating biharmonic coordinate...")
    W = compute_biharmonic(low_v_s, high_v, high_t)    
    # only surface
    W = W[high_J]

    calculated_high_v = np.dot(W, low_v_s)
    print(calculated_high_v.shape)

    high_poly = build_triangle_mesh(calculated_high_v, high_f)
    high_actor = make_actor(high_poly)
    high_actor.SetPosition(1, 0, 0)
    high_actor.GetProperty().SetColor(1,1,0)
    ren.AddActor(high_actor)


    #Gt High    
    high_poly1 = build_triangle_mesh(high_v_s, high_f)
    high_actor1 = make_actor(high_poly1)
    high_actor1.SetPosition(1, 1, 0)
    high_actor1.GetProperty().SetColor(1,1,0.5)
    ren.AddActor(high_actor1)



    ren.ResetCamera()
    renWin.Render()
    iren.Start()

