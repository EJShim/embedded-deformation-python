import vtk
from vtk.util import numpy_support
import autograd.numpy as np
# import numpy as npy

from autograd import grad 

import math
from numpy import linalg as la
from scipy import optimize


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
        self.n_point = self.pointBuffer.shape[0] 
        self.delta = 1e-6

        # self.polydata.ShallowCopy(polydata)
        self.pointActor = MakeSphereActor(self.polydata)
        self.actor = MakeActor(self.polydata)
        self.constraints = dict()
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

        # 12 * self.n_point is the problem dimension
        self.hess_manager = optimize.SR1(self.n_point*12, 'hess')



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
        self.constraints = dict()
        self.constraints[idx] = pos
        self.polydata.GetPoints().Modified()

    def calculateDeformationGraph(self, p, g_j, R_j, T_j):
        # Equation (1)
        p_hat = R_j@(p-g_j) + g_j - T_j
        return p_hat

    def getDeformedPosition(self, R, T):
        deformed_points = []
        # there should be a better than just looping through...
        for base_inx in range(self.n_point):
            p = self.pointBuffer[base_inx]
            # Equation (2)
            deformed_point = None
            for j, (R_j, T_j) in enumerate(zip(R, T)):
                g_j = self.pointBuffer[j]
                # Currently, for simplicity, considering all of weights are 1 equation (4)
                TMP_WEIGHT = 1
                # print("point add", p, g_j, R_j, T_j)
                if type(deformed_point) == type(None):
                    deformed_point = TMP_WEIGHT * self.calculateDeformationGraph(p, g_j, R_j, T_j)
                else:
                    deformed_point += TMP_WEIGHT * self.calculateDeformationGraph(p, g_j, R_j, T_j)  
            deformed_points.append(deformed_point)
        return deformed_points

    def objective_con(self, x):
        # Constraints defined as the following
        # {vert_inx(int): constraint poisiton(np.array)}
        R = x[:9*self.n_point].reshape(self.n_point, 3, 3)
        T = x[9*self.n_point:].reshape(self.n_point, 3)

        deformed_points = self.getDeformedPosition(R, T)

        # equation (8)
        final_loss = np.zeros(3)
        for vert_inx, pos_const in self.constraints.items():
            err = deformed_points[vert_inx] - pos_const
            final_loss += err
        
        return np.inner(final_loss,final_loss)



    # def calculateJacob(self, x_curr, func):
    #     delta_grad = []
    #     for i in range(len(x_curr)):
    #         x_delta = x_curr[i] + self.delta
    #         self.objective_con(self, constraints, R, T)


    def modified(self):        
        # inital guess equation (9)
        initial_guess = np.zeros(12*self.n_point)
        initial_guess_R = np.vstack([np.eye(3)]*self.n_point).reshape(-1)
        initial_guess[:9*self.n_point] = initial_guess_R

        del_inital_guess = initial_guess.copy() + self.delta
        # initial_guess_T = np.tile(np.zeros(3), (self.n_point, 1)).reshape(-1)
        # initial_guess = np.concat(initial_guess_R, initial_guess_T)

        # get what changed
        grad_objective = grad(self.objective_con)
        print("DELTA_X---------------------------------")
        # print(del_inital_guess - initial_guess)
        delta_x = del_inital_guess - initial_guess
        print("DELTA_GRAD---------------------------------")
        # print(grad_objective(del_inital_guess) - grad_objective(initial_guess))
        delta_grad = grad_objective(del_inital_guess) - grad_objective(initial_guess)
        self.hess_manager.update(delta_x, delta_grad)
        hess = self.hess_manager.get_matrix()





        from pdb import set_trace as st
        st()
        asdf
    



        prob_dim = initial_guess.shape[0]





        
        print(initial_guess.shape)
        asdf

        optimize




        # J = np.zeros((len(self.glyphs)*12, 1), dtype=np.float64)
        # f = np.zeros((len(self.glyphs)*12, 1), dtype=np.float64)

        # for j, glyph in enumerate(self.glyphs):
        #     matrix = glyph.GetUserMatrix() 
        #     print("i------------------")
        #     print(matrix)
        #     J[j*12+0] = matrix.GetElement(0, 0)
        #     J[j*12+1] = matrix.GetElement(1, 0)
        #     J[j*12+2] = matrix.GetElement(2, 0)

        #     J[j*12+3] = matrix.GetElement(0, 1)
        #     J[j*12+4] = matrix.GetElement(1, 1)
        #     J[j*12+5] = matrix.GetElement(2, 1)

        #     J[j*12+6] = matrix.GetElement(0, 2)
        #     J[j*12+7] = matrix.GetElement(1, 2)
        #     J[j*12+8] = matrix.GetElement(2, 2)

        #     box_pos = np.array([matrix.GetElement(0, 3), matrix.GetElement(1, 3), matrix.GetElement(2, 3)], dtype=np.float64 )

        #     J[j*12+9] = box_pos[0]
        #     J[j*12+10] = box_pos[1]
        #     J[j*12+11] = box_pos[2]

            
        #     # equation 1 make current position -> center
        #     # trans = self.pointBuffer[i] - box_pos
        #     # f[j*12+9] = trans[0] * 0.001
        #     # f[j*12+10] = trans[1]* 0.001
        #     # f[j*12+11] = trans[2]
        # asdf
        # print(i, ":", np.sum(f))
        
        # Jt = J.transpose()        
        # JtJ = np.matmul(J , Jt)
        # eye = np.identity(JtJ.shape[0])
        # JtJ += eye * 0.00000001 # Prevent Singular?
        # Jtf = -J*f
            
        
        # # # #Solve Dynamics
        # deltas = np.linalg.solve(JtJ, Jtf )        


        # updateJ = J + deltas
        # #Update Glyphs
        # for j, glyph in enumerate(self.glyphs):
        #     matrix = glyph.GetUserMatrix() 
        #     matrix.SetElement(0, 0, updateJ[j*12+0])
        #     matrix.SetElement(1, 0, updateJ[j*12+1])
        #     matrix.SetElement(2, 0, updateJ[j*12+2])

        #     matrix.SetElement(0, 1, updateJ[j*12+3])
        #     matrix.SetElement(1, 1, updateJ[j*12+4])
        #     matrix.SetElement(2, 1, updateJ[j*12+5])

        #     matrix.SetElement(0, 2, updateJ[j*12+6])
        #     matrix.SetElement(1, 2, updateJ[j*12+7])
        #     matrix.SetElement(2, 2, updateJ[j*12+8])

        #     matrix.SetElement(0, 3, updateJ[j*12+9])
        #     matrix.SetElement(1, 3, updateJ[j*12+10])
        #     matrix.SetElement(2, 3, updateJ[j*12+11])

        #     #Update Point Position Too
            

    #untill converge