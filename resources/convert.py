import vtk


reader = vtk.vtkPLYReader()
reader.SetFileName("sample_faust.ply")

writer = vtk.vtkOBJWriter()
writer.SetInputConnection(reader.GetOutputPort())
writer.SetFileName("sample_faust.obj")
writer.Write()