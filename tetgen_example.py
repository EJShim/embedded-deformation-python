import vtk
import igl
from vtk.util import numpy_support

import pyvista as pv
import tetgen
import numpy as np
pv.set_plot_theme('document')

def make_actor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

if __name__ == "__main__":

    pv.set_plot_theme('document')


    reader = vtk.vtkOBJReader()
    reader.SetFileName("resources/sample_faust.obj")
    reader.Update()

    # input polydata
    polydata = reader.GetOutput()

    print(polydata.GetNumberOfPoints())
    print(polydata.GetNumberOfPolys())
    

    # Make pyvista polydata
    pol = pv.PolyData(polydata)
    tet = tetgen.TetGen(pol)
    tet.make_manifold()
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)    

    print("================================")
    print(tet.mesh.GetNumberOfPoints())
    print(tet.mesh.GetNumberOfPolys())


    print("================================")
    print(tet.grid.GetNumberOfPoints())
    print(tet.grid.GetNumberOfCells())



    tet_v = numpy_support.vtk_to_numpy( tet.grid.GetPoints().GetData() )
    tet_t =  numpy_support.vtk_to_numpy( tet.grid.GetCells().GetData() )
    tet_t = np.reshape(tet_t, (tet_t.shape[0] // 5, 5))[:,1:]

    sur_v = numpy_support.vtk_to_numpy( tet.mesh.GetPoints().GetData() )
    sur_f =  numpy_support.vtk_to_numpy( tet.mesh.GetPolys().GetData() )
    sur_f = np.reshape(sur_f, (sur_f.shape[0] // 4, 4))[:,1:]

    # Validate IGL Extract Surface Index
    _,_,_, J = igl.remove_unreferenced(tet_v,sur_f)

    print(J.shape)
    print(sur_v.shape)
    print(tet_v.shape)





    exit()
    
    grid = tet.grid
    grid.plot(show_edges=True)
    

    # get cell centroids
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 1] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)

    # advanced plotting
    plotter = pv.Plotter()
    plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
    plotter.add_mesh(pol, 'r', 'wireframe')
    plotter.add_legend([[' Input Mesh ', 'r'],
                        [' Tessellated Mesh ', 'black']])
    plotter.show()
    exit()
    # tetgeen polydata


    # polydata decimation
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(0.75)
    decimate.VolumePreservationOff()
    decimate.Update()
    
    dec_polydata = decimate.GetOutput()
    dec_actor = make_actor(dec_polydata)
    dec_actor.SetPosition(1,0,0)
    ren.AddActor(dec_actor)


    ren.ResetCamera()
    renWin.Render()
    iren.Start()

