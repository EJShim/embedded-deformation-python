import vtk
from vtk.util import numpy_support
import numpy as np
import math


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
        self.pointBuffer = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

        # self.polydata.ShallowCopy(polydata)
        self.pointActor = MakeSphereActor(self.polydata)
        self.actor = MakeActor(self.polydata)

        self.glyphs = []
        for i in range(self.polydata.GetNumberOfPoints()):
            glyph = MakeBoxGlyph()
            pos = self.polydata.GetPoints().GetPoint(i)
            matrix = vtk.vtkMatrix4x4()
            matrix.SetElement(0, 3, pos[0])
            matrix.SetElement(1, 3, pos[1])
            matrix.SetElement(2, 3, pos[2])
            glyph.SetUserMatrix(matrix)
            
            self.glyphs.append(glyph)


    def addToRenderer(self, renderer):
        renderer.AddActor(self.pointActor)
        renderer.AddActor(self.actor)
        
        for glyph in self.glyphs:
            renderer.AddActor(glyph)


    def updatePoint(self, idx, pos):        
        # self.polydata.GetPoints().SetPoint(idx, pos[0], pos[1], pos[2])
        self.pointBuffer[idx][0] = pos[0]
        self.pointBuffer[idx][1] = pos[1]
        self.pointBuffer[idx][2] = pos[2]
        self.polydata.GetPoints().Modified()

    def modified(self):        


        for i in range(10):
            J = np.zeros((len(self.glyphs)*12, 1), dtype=np.float64)
            f = np.zeros((len(self.glyphs)*12, 1), dtype=np.float64)
            for j, glyph in enumerate(self.glyphs):
                matrix = glyph.GetUserMatrix() 
                J[j*12+0] = matrix.GetElement(0, 0)
                J[j*12+1] = matrix.GetElement(1, 0)
                J[j*12+2] = matrix.GetElement(2, 0)

                J[j*12+3] = matrix.GetElement(0, 1)
                J[j*12+4] = matrix.GetElement(1, 1)
                J[j*12+5] = matrix.GetElement(2, 1)

                J[j*12+6] = matrix.GetElement(0, 2)
                J[j*12+7] = matrix.GetElement(1, 2)
                J[j*12+8] = matrix.GetElement(2, 2)

                pos = np.array([matrix.GetElement(0, 3), matrix.GetElement(1, 3), matrix.GetElement(2, 3)], dtype=np.float64 )
                J[j*12+9] = pos[0]
                J[j*12+10] = pos[1]
                J[j*12+11] = pos[2]


                #Econ???
                trans = np.array(self.polydata.GetPoint(j)) - pos
                f[j*12+9] = trans[0] * 0.001
                f[j*12+10] = trans[1]* 0.001
                f[j*12+11] = trans[2]
            
            print(i, ":", np.sum(f))
            
            Jt = J.transpose()        
            JtJ = np.matmul(J , Jt)
            eye = np.identity(JtJ.shape[0])
            JtJ += eye * 0.00000001 # Prevent Singular?
            Jtf = -J*f
                
            
            # # #Solve Dynamics
            deltas = np.linalg.solve(JtJ, Jtf )        


            updateJ = J + deltas
            #Update Glyphs
            for j, glyph in enumerate(self.glyphs):
                matrix = glyph.GetUserMatrix() 
                matrix.SetElement(0, 0, updateJ[j*12+0])
                matrix.SetElement(1, 0, updateJ[j*12+1])
                matrix.SetElement(2, 0, updateJ[j*12+2])

                matrix.SetElement(0, 1, updateJ[j*12+3])
                matrix.SetElement(1, 1, updateJ[j*12+4])
                matrix.SetElement(2, 1, updateJ[j*12+5])

                matrix.SetElement(0, 2, updateJ[j*12+6])
                matrix.SetElement(1, 2, updateJ[j*12+7])
                matrix.SetElement(2, 2, updateJ[j*12+8])

                matrix.SetElement(0, 3, updateJ[j*12+9])
                matrix.SetElement(1, 3, updateJ[j*12+10])
                matrix.SetElement(2, 3, updateJ[j*12+11])

                #Update Point Position Too
                

        #untill converge