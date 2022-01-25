from distutils.log import error
import vtk
from vtk.util import numpy_support
import numpy as np
# import numpy as npy

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
        self.constraints = {i:self.pointBuffer[i] for i in range(self.n_point)}
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
        
        # define neighborhood

        self.neighbor = dict()
        for i in range(self.polydata.GetNumberOfPoints()):
            if i == 0:
                self.neighbor[i] = [1]
            elif i == self.polydata.GetNumberOfPoints() - 1:
                self.neighbor[i] = [i-1]
            else:
                self.neighbor[i] = [i-1,i+1]

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
        self.constraints[idx] = pos
        self.polydata.GetPoints().Modified()

    def calculateDeformationGraph(self, p, g_j, R_j, T_j):
        # Equation (1)
        p_hat = R_j@(p-g_j) + g_j - T_j
        return p_hat

    def getDeformedPosition(self, R, T):
        deformed_points = []
        # calculate weights

        # there should be a better than just looping through...
        for base_inx in range(self.n_point):

            p = self.pointBuffer[base_inx]
            # Equation (2)
            deformed_point = None

            # Currently, for simplicity, considering all of weights are considered (4)
            err_dist = np.linalg.norm((p - self.pointBuffer), axis=1)
            w_j_v_i = (1-err_dist/max(err_dist))**2

            for j, (R_j, T_j) in enumerate(zip(R, T)):
                g_j = self.pointBuffer[j]

                # Equation (1)
                if type(deformed_point) == type(None):
                    deformed_point = w_j_v_i[j] * self.calculateDeformationGraph(p, g_j, R_j, T_j)
                else:
                    each_deform = w_j_v_i[j] * self.calculateDeformationGraph(p, g_j, R_j, T_j)
                    deformed_point += each_deform

            deformed_points.append(deformed_point)
        return deformed_points

    def objective_con(self, x):
        # Constraints defined as the following
        # {vert_inx(int): constraint poisiton(np.array)}
        R = x[:9*self.n_point].reshape(self.n_point, 3, 3)
        T = x[9*self.n_point:].reshape(self.n_point, 3)

        deformed_points = self.getDeformedPosition(R, T)

        # equation (8)
        final_loss = 0
        for vert_inx, pos_const in self.constraints.items():

            err = deformed_points[vert_inx] - pos_const
            # print("vert_inx", vert_inx)
            # print("deformed_points[vert_inx]", deformed_points[vert_inx])
            # print("pos_const", pos_const)
            final_loss += np.inner(err,err)
        
        return final_loss

    def objective_reg(self, x):

        # def calculateDeformationGraph(self, p, g_j, R_j, T_j):
        # # Equation (1)
        # p_hat = R_j@(p-g_j) + g_j - T_j
        # return p_hat



        # equation (7)
        alpha = 1
        final_loss = 0

        R = x[:9*self.n_point].reshape(self.n_point, 3, 3)
        T = x[9*self.n_point:].reshape(self.n_point, 3)

        for j, (R_j, T_j) in enumerate(zip(R, T)):


            neighbor = self.neighbor[j]
            g_j = self.pointBuffer[j]
            for k in neighbor:
                # from pdb import set_trace as st
                # st()
                g_k = self.pointBuffer[k]
                t_k = T[k]
                deformed_position = R_j@(g_k-g_j)+g_j+T_j
                actual_position = g_k+t_k

                final_loss += alpha*np.linalg.norm(deformed_position-actual_position)
        return final_loss

    def objective_rot(self, x):
        R = x[:9*self.n_point].reshape(self.n_point, 3, 3)
        # equation (5)
        final_loss = 0
        for R_j in R:
            c1, c2, c3 = R_j[:, 0], R_j[:, 1], R_j[:, 2]
            final_loss += np.inner(c1, c2)**2 + np.inner(c2, c3)**2 + np.inner(c2, c3)**2 + (np.inner(c1, c1)-1)**2+ (np.inner(c2, c2)-1)**2+ (np.inner(c3, c3)-1)**2
        return final_loss

    def objective(self, x, w_rot, w_reg, w_con):
        # print("-----x", x)
        # return w_rot*self.objective_rot(x) + w_reg*self.objective_reg(x) + w_con*self.objective_con(x)
        return w_rot*self.objective_rot(x) + w_con*self.objective_con(x)


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

        import time
        start = time.time()
        w_rot, w_reg, w_con = 10, 25, 100
        val = optimize.minimize(self.objective, initial_guess, args=(w_rot, w_reg, w_con), method='BFGS', tol=1e-3)
        end = time.time()
        print("optimization time:", end - start)
        
        deformed_R_lst = val.x[:9*self.n_point].reshape(self.n_point, 3, 3)
        deformed_T_lst = val.x[9*self.n_point:].reshape(self.n_point, 3)
        # print(deformed_R_lst)
        # from pdb import set_trace as st
        # st()
        # # print("point_BUFFER-----",self.pointBuffer)
        # print("OPTIMIZED VALUE-----",self.getDeformedPosition(deformed_R_lst, deformed_T_lst))
        # # for j, (R_j, T_j) in enumerate(zip(deformed_R_lst, deformed_T_lst)):
        # #     print("point",j, self.pointBuffer[j])
        # #     print("ROATION------")
        # #     print(R_j)
        # #     print("TRANSLATION------")
        # #     print(T_j)
        # #     print("DEFORMED...")
        # #     pos = self.polydata.GetPoints().GetPoint(j)

        # # self.objective_con(initial_guess)


        # print(deformed_R_lst.shape)
        # print(deformed_T_lst.shape)


        deformed_position = self.getDeformedPosition(deformed_R_lst, deformed_T_lst)

        #Update Glyphs
        for j, glyph in enumerate(self.glyphs):
            
            matrix = glyph.GetUserMatrix() 
            matrix.SetElement(0, 0, deformed_R_lst[j, 0, 0])
            matrix.SetElement(1, 0, deformed_R_lst[j, 0, 1])
            matrix.SetElement(2, 0, deformed_R_lst[j, 0, 2])

            matrix.SetElement(0, 1, deformed_R_lst[j, 1, 0])
            matrix.SetElement(1, 1, deformed_R_lst[j, 1, 1])
            matrix.SetElement(2, 1, deformed_R_lst[j, 1, 2])

            matrix.SetElement(0, 2, deformed_R_lst[j, 2, 0])
            matrix.SetElement(1, 2, deformed_R_lst[j, 2, 1])
            matrix.SetElement(2, 2, deformed_R_lst[j, 2, 2])
            # self.pointBuffer[j] += deformed_T_lst[j]
            matrix.SetElement(0, 3, deformed_position[j][0])
            matrix.SetElement(1, 3, deformed_position[j][1])
            matrix.SetElement(2, 3, deformed_position[j][2])
 
            #Update Point Position Too
            

    #untill converge