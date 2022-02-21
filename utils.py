from multiprocessing.sharedctypes import Value
from typing import final
import vtk
from vtk.util import numpy_support
import numpy as np
import math
from scipy.linalg import cho_factor, cho_solve


# THIS CODE IS COMING FROM https://github.com/Hunger720/Embedded_Deformation_for_Shape_Manipulation/blob/master
W_ROT = 15
W_REG = 20
W_CON = 100

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
    
def userMat2RG(user_matrix):
    ar = np.zeros((4,4))

    user_matrix.DeepCopy(ar.ravel(), user_matrix)
    R = ar[:3,:3]
    g = ar[:3, 3]

    return R, g

class DeformableGraph():
    def __init__(self, polydata, k_nearest=3, tol=1e-6):        
        
        self.polydata = polydata
        self.pointBuffer = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        self.n_node = len(self.pointBuffer)

        # self.polydata.ShallowCopy(polydata)
        self.pointActor = MakeSphereActor(self.polydata)
        self.actor = MakeActor(self.polydata)
        self.constraints = dict()#{i:self.pointBuffer[i] for i in range(self.n_node)}
        self.constraints[0] = self.pointBuffer[0]
        self.glyphs = []
        self.tol = tol
        self.k_nearest = k_nearest

        for i in range(self.polydata.GetNumberOfPoints()):
            glyph = MakeBoxGlyph()
            pos = self.polydata.GetPoints().GetPoint(i)
            matrix = vtk.vtkMatrix4x4()
            matrix.SetElement(0, 3, pos[0])
            matrix.SetElement(1, 3, pos[1])
            matrix.SetElement(2, 3, pos[2])
            glyph.SetUserMatrix(matrix)
            
            self.glyphs.append(glyph)
    
    def initalize(self):
        # initial values
        self.rot = np.concatenate([np.eye(3)]*self.n_node).reshape(self.n_node, -1)
        self.trans = np.zeros((self.n_node,3))

        node = []
        # node position
        for i in range(self.n_node):
            mat = self.glyphs[i].GetUserMatrix()
            _, G = userMat2RG(mat)
            node.append(G)
        self.node = node


        # node connectivity
        edges = np.zeros((self.n_node,self.n_node)).astype(bool)
        for i in range(self.n_node):

            k_nearest = self.findNearestNodes(i)

            for j in range(self.k_nearest):
                # don't consider node itself.
                for jj in range(j+1, self.k_nearest):
                    edges[k_nearest[jj]][k_nearest[j]] = True
                    edges[k_nearest[j]][k_nearest[jj]] = True
        self.edges = edges

        # number of edge
        n_edge = 0
        for row in self.edges:
            for e in row:
                if e: n_edge += 1
        n_edge /= 2
        self.n_edge = n_edge

    def updatePoint(self, idx, pos):        
        self.pointBuffer[idx][0] = pos[0]
        self.pointBuffer[idx][1] = pos[1]
        self.pointBuffer[idx][2] = pos[2]
        self.constraints[idx] = pos
        self.polydata.GetPoints().Modified()

    def findNearestNodes(self, base_inx):
        dist = []
        for j in range(self.n_node):
            # TODO: Since we have just vertex we can just use node index
            dist.append(np.linalg.norm(self.node[base_inx] - self.node[j], ord=2))
        
        sorted_dist = np.argsort(dist)

        return sorted_dist[:self.k_nearest]


    def computeWeights(self, vertex):
        # Equation (4)
        # compute k+1 nearestNode(don't consider the node itself)
        # return {"nearest_inx":"weight"}
        dist = np.linalg.norm((vertex - self.node), axis=1)
        arg_dist = np.argsort(dist)[:self.k_nearest+1]
        sorted_dist = dist[arg_dist]
        max_dist = sorted_dist[-1]

        w_j_v_i = (1-sorted_dist/max_dist)**2
        # normalize to 1
        w_j_v_i /= np.linalg.norm(w_j_v_i, ord=1)
        weights = dict()
        for i, dist_inx in enumerate(arg_dist):
            weights[dist_inx] = w_j_v_i[i]
        return weights

    def getDeformedPosition(self, vertex_inx):
        pos = np.zeros(3)
        # calculate weights
        # TODO: currently, we only have node; so we can say vertex_inx == node_inx
        if True:
            vertex_pos = self.node[vertex_inx]
        else:
            raise NotImplementedError("vertex_pos = self.point_inx[vertex_inx]")

        weight_info = self.computeWeights(vertex_pos)
        for inx, weight in weight_info.items():
            tmp = vertex_pos - self.node[inx]
            # Equation (1)
            # for i in range(3):
            #     pos[i] += weight*(self.rot[inx][i]*tmp[0] + self.rot[inx][i+3]*tmp[1] + self.rot[inx][i+6]*tmp[2]+self.node[inx][i]+self.trans[inx][i])
            pos += weight * (self.rot[inx].reshape(3,3).T@tmp + self.node[inx]+self.trans[inx])
            # if np.allclose(pos, pos_copy):
            #     print("POS1",pos)
            #     print("POS2", pos_copy)
            #     print("-------------------------")
            # else:
            #     from pdb import set_trace as st
            #     st()
        return pos

    def objective_con(self):
        # Constraints defined as the following
        # {vert_inx(int): constraint poisiton(np.array)}


        # equation (8)
        final_loss = 0

        for vert_inx, pos_const in self.constraints.items():

            deformed_points = self.getDeformedPosition(vert_inx)
            err = deformed_points - pos_const
            final_loss += np.inner(err,err)
        return final_loss

    def objective_reg(self):
        alpha = 1
        final_loss = 0

        for j in range(self.n_node):
            for n in range(self.n_node):
                if self.edges[j][n]:
                    # equation (7)
                    err = self.rot[j].reshape(3,3).T@(self.node[n] - self.node[j]) + self.node[j] + self.trans[j] - (self.node[n] + self.trans[n])
                    final_loss += np.inner(err, err)


        return final_loss

    def objective_rot(self):
        # equation (5)
        final_loss = 0
        for j in range(self.n_node):
            R_j = self.rot[j].reshape(3,3)
            c1, c2, c3 = R_j[:, 0], R_j[:, 1], R_j[:, 2]
            final_loss += np.inner(c1, c2)**2 + np.inner(c2, c3)**2 + np.inner(c2, c3)**2 + (np.inner(c1, c1)-1)**2+ (np.inner(c2, c2)-1)**2+ (np.inner(c3, c3)-1)**2
    

        return final_loss

    def computeF(self):

        fx = np.zeros(int(self.n_node*6+self.n_edge*6+3*len(self.constraints)))
        inx = 0

        # # rot
        for j in range(self.n_node):
            fx[inx] = (self.rot[j][0]*self.rot[j][3]+self.rot[j][1]*self.rot[j][4]+self.rot[j][2]*self.rot[j][5])*np.sqrt(W_ROT); inx += 1
            fx[inx] = (self.rot[j][0]*self.rot[j][6]+self.rot[j][1]*self.rot[j][7]+self.rot[j][2]*self.rot[j][8])*np.sqrt(W_ROT); inx += 1
            fx[inx] = (self.rot[j][3]*self.rot[j][6]+self.rot[j][4]*self.rot[j][7]+self.rot[j][5]*self.rot[j][8])*np.sqrt(W_ROT); inx += 1
            fx[inx] = (self.rot[j][0]*self.rot[j][0]+self.rot[j][1]*self.rot[j][1]+self.rot[j][2]*self.rot[j][2]-1)*np.sqrt(W_ROT); inx += 1
            fx[inx] = (self.rot[j][3]*self.rot[j][3]+self.rot[j][4]*self.rot[j][4]+self.rot[j][5]*self.rot[j][5]-1)*np.sqrt(W_ROT); inx += 1
            fx[inx] = (self.rot[j][6]*self.rot[j][6]+self.rot[j][7]*self.rot[j][7]+self.rot[j][8]*self.rot[j][8]-1)*np.sqrt(W_ROT); inx += 1

        # reg
        for j in range(self.n_node):
            for k in range(self.n_node):
                if(self.edges[j][k]):
                    for i in range(3):
                        fx[inx] = (self.rot[j][i]*(self.node[k][0]-self.node[j][0])
                                +self.rot[j][i+3]*(self.node[k][1]-self.node[j][1])
                                +self.rot[j][i+6]*(self.node[k][2]-self.node[j][2])
                                +self.node[j][i]+self.trans[j][i]-self.node[k][i]-self.trans[k][i])*np.sqrt(W_REG); inx += 1

        ###########################3
        # debug
        ###############################
        # inx = 84
        # con
        for vert_inx, pos_con in self.constraints.items():
            # TODO every vertex is just a node.

            deformed_v = self.getDeformedPosition(vert_inx)
            for i in range(3):
                fx[inx] = (deformed_v[i] - pos_con[i]) * np.sqrt(W_CON); inx += 1
        return fx

    def computeJ(self):
        Jacobi = np.zeros((int(self.n_node*6+self.n_edge*6+3*len(self.constraints)), 12*self.n_node))
        inx = 0

        # rot
        for j in range(self.n_node): 
            Jacobi[inx,0+12*j] = self.rot[j][3]*np.sqrt(W_ROT);Jacobi[inx,1+12*j] = self.rot[j][4]*np.sqrt(W_ROT);Jacobi[inx,2+12*j] = self.rot[j][5]*np.sqrt(W_ROT)
            Jacobi[inx,3+12*j] = self.rot[j][0]*np.sqrt(W_ROT);Jacobi[inx,4+12*j] = self.rot[j][1]*np.sqrt(W_ROT);Jacobi[inx,5+12*j] = self.rot[j][2]*np.sqrt(W_ROT);inx+=1

            Jacobi[inx,0+12*j] = self.rot[j][6]*np.sqrt(W_ROT);Jacobi[inx,1+12*j] = self.rot[j][7]*np.sqrt(W_ROT);Jacobi[inx,2+12*j] = self.rot[j][8]*np.sqrt(W_ROT)
            Jacobi[inx,6+12*j] = self.rot[j][0]*np.sqrt(W_ROT);Jacobi[inx,7+12*j] = self.rot[j][1]*np.sqrt(W_ROT);Jacobi[inx,8+12*j] = self.rot[j][2]*np.sqrt(W_ROT);inx+=1

            Jacobi[inx,3+12*j] = self.rot[j][6]*np.sqrt(W_ROT);Jacobi[inx,4+12*j] = self.rot[j][7]*np.sqrt(W_ROT);Jacobi[inx,5+12*j] = self.rot[j][8]*np.sqrt(W_ROT)
            Jacobi[inx,6+12*j] = self.rot[j][3]*np.sqrt(W_ROT);Jacobi[inx,7+12*j] = self.rot[j][4]*np.sqrt(W_ROT);Jacobi[inx,8+12*j] = self.rot[j][5]*np.sqrt(W_ROT);inx+=1

            for i in range(9):
                if i == 3 or i == 6: inx += 1
                Jacobi[inx, i+12*j] = 2*self.rot[j][i]*np.sqrt(W_ROT)

            inx += 1
        # reg
        for j in range(self.n_node):
            for k in range(self.n_node):
                if self.edges[j][k]:
                    for i in range(3):
                        for ii in range(3): 
                            Jacobi[inx,12*j+3*ii+i] = (self.node[k][ii]-self.node[j][ii])*np.sqrt(W_REG)
                        Jacobi[inx,12*j+i+9] = np.sqrt(W_REG)
                        Jacobi[inx,12*k+i+9] = -np.sqrt(W_REG)
                        inx+=1
        # inx = 84

        # Econ
        for vert_inx, pos_con in self.constraints.items():

            weight_info = self.computeWeights(self.node[vert_inx])
            for i in range(3):
                for vert_inx, weight in weight_info.items():
                    for ii in range(3):

                        Jacobi[inx, ii*3+i+12*vert_inx] = weight*(pos_con[ii]-self.node[vert_inx][ii])*np.sqrt(W_CON)
                    Jacobi[inx, 9+i+12*vert_inx] = weight*np.sqrt(W_CON)
                inx += 1
        return Jacobi


    def objective(self, x):

        self.updateMat(x)


        
        L_rot = self.objective_rot()
        L_reg = self.objective_reg()
        L_con = self.objective_con()
        print(f"L_rot: {L_rot}, L_reg: {L_reg}, L_con: {L_con}")
        return W_ROT*L_rot + W_REG*L_reg + W_CON*L_con
    
        L_rot = self.objective_rot()
        L_con = self.objective_con()
        print(f"L_rot: {L_rot}, L_con: {L_con}")

        return W_ROT*L_rot + W_CON*L_con
        # return L_con

    def updateMat(self, x):
        for j, glyph in enumerate(self.glyphs):
            matrix = glyph.GetUserMatrix() 
            matrix.SetElement(0, 0, x[j*12+0])
            matrix.SetElement(1, 0, x[j*12+1])
            matrix.SetElement(2, 0, x[j*12+2])

            matrix.SetElement(0, 1, x[j*12+3])
            matrix.SetElement(1, 1, x[j*12+4])
            matrix.SetElement(2, 1, x[j*12+5])

            matrix.SetElement(0, 2, x[j*12+6])
            matrix.SetElement(1, 2, x[j*12+7])
            matrix.SetElement(2, 2, x[j*12+8])

            for i in range(9):
                self.rot[j][i] = x[j*12+i]
            for i in range(3):
                self.trans[j][i] = x[j*12+9+i]
            pos = self.getDeformedPosition(j)
            print("NODE", self.node[j], pos)
            matrix.SetElement(0, 3, pos[0])
            matrix.SetElement(1, 3, pos[1])
            matrix.SetElement(2, 3, pos[2])


    def modified(self):        
        self.initalize()

        x = np.zeros(12*self.n_node)

        # update to current
        for j in range(self.n_node):
            x[j*12:j*12+9] = self.rot[j].reshape(-1)
            x[j*12+9:j*12+12] = self.trans[j]

        obj, obj_updated = 0, self.objective(x)
        prev_position = x.copy()

        for i in range(20):
            if abs(obj_updated - obj) < self.tol:
                print("optimization finished!")
                break

            obj = obj_updated
            delta = self.computeF()
            # 6*dg.n_nodes+6*dg.n_edges+3*p x 12*self.n_nodes
            J = self.computeJ() # 
            JtJ = J.T@J

            # prevent singulaity
            JtJ += 1e-6*np.eye(12*self.n_node)
            mat, low = cho_factor(JtJ)
            updated_delta = cho_solve((mat, low), -J.T@delta)
            x += updated_delta
            obj_updated = self.objective(x)
        
        # it is inefficient.
        for j in range(len(self.glyphs)):
            pos = self.getDeformedPosition(j)
            self.node[j] = pos
            self.updatePoint(j, pos)
        
        # for i, (pos_prev, pos_curr) in enumerate(zip(prev_position.reshape(self.n_node,-1), x.reshape(self.n_node,-1))):        
        #     print(f"{i} Node", self.node[i])
        #     print("prev:",pos_prev[9:],"\n","curr",pos_curr[9:])


        # now change position


    def addToRenderer(self, renderer):
        renderer.AddActor(self.pointActor)
        renderer.AddActor(self.actor)
        
        for glyph in self.glyphs:
            renderer.AddActor(glyph)