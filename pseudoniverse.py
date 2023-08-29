import vtk
import igl
import math
import argparse
import numpy as np
from vtk.util import numpy_support
from mesh_sampling import read_polydata, make_actor, make_point_actor, build_polydata
from mesh_sampling import Mesh
from mesh_sampling import generate_transform_matrices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Generate Transform Matrix')
    parser.add_argument('--template', type = str,  default="resources/faust_manifold.obj")
    parser.add_argument("--output", type=str, default = "transform")
    args = parser.parse_args()


    #Read Template
    templatePoly = read_polydata(args.template)

    mesh = Mesh()
    mesh.SetPolyData(templatePoly)


    ds_factors = [4,4,4,4]
    D, U, F, V, M = generate_transform_matrices(mesh, ds_factors, method='planarquadric')



    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)

    bounds = templatePoly.GetBounds()
    xlen = bounds[1] - bounds[0]
    ylen = bounds[3] - bounds[2]
    zlen = bounds[5] - bounds[4]

    # T arget
    actor = make_actor(M[0].polydata)
    ren.AddActor(actor)
    
    
    for idx, mesh in enumerate(M):
        actor = make_actor(mesh.polydata)
        actor.SetPosition(idx*xlen, ylen, 0)
        ren.AddActor(actor)

    # Appl yTransform
    v = M[0].v
    p_U = []
    for idx, down in enumerate(D):
        
        down_mat = down.todense()
        v = np.dot(down_mat, v)
        f = F[idx+1]
        polydata = build_polydata(v, f)
        actor = make_actor(polydata)
        actor.SetPosition(xlen * (idx+1), 0, 0)
        ren.AddActor(actor)

        # TODO : try to calculate pseudoinverse
        up_mat = np.linalg.pinv(down_mat)
        p_U.append(up_mat)

    p_v = v.copy()
    for idx, up in reversed(list(enumerate(U))):
        v = np.dot(up.todense(), v)
        f = F[idx]
        polydata = build_polydata(v, f)
        actor = make_actor(polydata)
        actor.SetPosition(xlen*len(D) + xlen * (len(U)-idx),0, 0)
        ren.AddActor(actor)

        p_up = p_U[idx]
        p_v = np.dot(p_up, p_v)
        p_polydata = build_polydata(p_v, f)
        p_actor = make_actor(p_polydata)
        p_actor.SetPosition(xlen*len(D) + xlen * (len(U)-idx), -ylen, 0)
        ren.AddActor(p_actor)

        # print(up.shape, p_up.shape)

    ren.ResetCamera()
    renWin.Render()
    iren.Start()