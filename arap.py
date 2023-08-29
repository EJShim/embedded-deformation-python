import vtk

def make_actor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


class InteractionController:
    def __init__(self, renWin):

        self.mode = False
        self.renWin = renWin
        self.ren = renWin.GetRenderers().GetFirstRenderer() 
        self.ren.SetGradientBackground(True)
        
        self.target = None

        self.timer_id = -1
    
        self.updateMode()

    def SetTarget(self, actor):
        self.target = actor
        polydata = actor.GetMapper().GetInput()         

        # Add Interactor Style only when the target exists
        renWin.GetInteractor().GetInteractorStyle().AddObserver("LeftButtonPressEvent", self.leftButtonDown)
        renWin.GetInteractor().GetInteractorStyle().AddObserver("KeyPressEvent", self.keyDown)
    
    def leftButtonDown(self, interactor, e):
        
        if self.mode : # Simulation mode
            print("TODO : Pick control point")
        else: # Control point mode
            print("TODO : Add Control Point")
        
        # superclass
        interactor.OnLeftButtonDown()

    def keyDown(self, interactor, e):
        
        keycode = interactor.GetInteractor().GetKeySym() 
        if keycode == "space":
            self.mode = not self.mode
            self.updateMode()
    
    def updateMode(self):

        if self.mode: # simulation mode
            self.ren.SetBackground(.5,.5,.9)
            self.ren.SetBackground2(.9,.9,.9)

            self.timer_id = self.renWin.GetInteractor().CreateRepeatingTimer(10)   
        else:
            self.ren.SetBackground(.2,.2,.2)
            self.ren.SetBackground2(.9,.9,.9)
        

        self.renWin.Render()




if __name__ == "__main__":

    
    iren = vtk.vtkRenderWindowInteractor()
    interactorStyle =  vtk.vtkInteractorStyleTrackballCamera()        
    iren.SetInteractorStyle(interactorStyle)
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000, 1000)
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)

    # Add controller
    controller = InteractionController(renWin)

 
    reader = vtk.vtkOBJReader()
    reader.SetFileName("resources/decimated-knight.obj")
    reader.Update()

    polydata = reader.GetOutput()
    polydata.GetPointData().RemoveArray("Normals")
    actor = make_actor(polydata)

    ren.AddActor(actor)    


    # Add Target Polydata    
    controller.SetTarget(actor)
    # controller.SetPolyData(polydata)






    ren.ResetCamera()
    renWin.Render()
    iren.Start()
    