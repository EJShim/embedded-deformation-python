import vtk


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
    


class DeformableGraph():
    def __init__(self, polydata):        
        
        self.polydata = polydata
        
        # self.polydata.ShallowCopy(polydata)
        self.pointActor = MakeSphereActor(self.polydata)
        self.actor = MakeActor(self.polydata)

        self.glyphs = []
        for i in range(self.polydata.GetNumberOfPoints()):
            glyph = MakeBoxGlyph()
            glyph.SetPosition(self.polydata.GetPoints().GetPoint(i))
            self.glyphs.append(glyph)


    def addToRenderer(self, renderer):
        renderer.AddActor(self.pointActor)
        renderer.AddActor(self.actor)
        
        for glyph in self.glyphs:
            renderer.AddActor(glyph)


    def updatePoint(self, idx, pos):        
        self.polydata.GetPoints().SetPoint(idx, pos[0], pos[1], pos[2])
        self.polydata.GetPoints().Modified()

    def modified(self):
        print("Solve Dynamics!, update Glyphs")