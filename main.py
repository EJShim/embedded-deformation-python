import vtk
import igl
from ref import build_tet_mesh, build_triangle_mesh, make_actor
from ref import compute_biharmonic
import numpy as np
from vtk.util import numpy_support


def make_tet_actor(tet):
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(tet)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

if __name__ == "__main__":
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    ren.SetBackground(1,1,1)
    renWin.AddRenderer(ren)

    # read tet octopus
    Vj, t, f = igl.read_mesh("resources/octopus-high.mesh")
    _,_,_,J = igl.remove_unreferenced(Vj, f)
    v = Vj[J]
    
    # render original array
    polydata = build_triangle_mesh(v, f)
    actor = make_actor(polydata)
    ren.AddActor(actor)


    # apply decimation
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(0.99)
    decimate.VolumePreservationOff()
    decimate.Update()        
    low_polydata = decimate.GetOutput()
    Vl = numpy_support.vtk_to_numpy( low_polydata.GetPoints().GetData())
    low_actor = make_actor(low_polydata)
    low_actor.SetPosition(1,0,0)
    ren.AddActor(low_actor)
    print("Decimation : ", v.shape,  Vl.shape)
    
    
    # TODO: make_tetrahedr Vj, t
    Vj, t
    tet = build_tet_mesh(Vj, t)
    
    clipPlane = vtk.vtkPlane()
    clipPlane.SetOrigin(tet.GetCenter())
    clipPlane.SetNormal([-1.0, -1.0, 1.0])
    clipper = vtk.vtkClipDataSet()
    clipper.SetClipFunction(clipPlane)
    clipper.SetInputData(tet)
    clipper.SetValue(0.0)
    clipper.GenerateClippedOutputOn()
    clipper.Update()
    tet_actor = make_tet_actor(clipper.GetOutput())
    tet_actor.SetPosition(1, -1, 0)
    tet_actor.GetProperty().SetEdgeVisibility(True)
    ren.AddActor(tet_actor)




    # print("Calculating...")
    # Wj = compute_biharmonic(Vl, Vj, t)
    # W = Wj[J]

    # # Reconstruction
    # print(W.shape)
    # Vr = np.matmul(W, Vl)
    # recon_poly = build_triangle_mesh(Vr, f)
    # recon_actor = make_actor(recon_poly)
    # recon_actor.SetPosition(2, 0, 0)
    # recon_actor.GetProperty().SetColor(1.0, 0.9, 0.9)
    # ren.AddActor(recon_actor)

    




    ren.ResetCamera()
    renWin.Render()
    iren.Start()