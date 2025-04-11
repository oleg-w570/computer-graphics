import itk
import vtk

def main():
    image_path = "niivue-images/CT_Philips.nii.gz"    
    
    itk_image = itk.imread(image_path)
    itk_image = itk.median_image_filter(itk_image)
    itk_image = itk.gradient_magnitude_recursive_gaussian_image_filter(itk_image)

    vtk_image = itk.vtk_image_from_image(itk_image)
    
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_image)
    mc.SetValue(0, 100)
    mc.Update()
    vtk_image = mc.GetOutput()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(vtk_image)
    smoother.SetNumberOfIterations(15)
    smoother.SetRelaxationFactor(0.1)
    smoother.Update()
    vtk_image = smoother.GetOutput()

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(vtk_image)
    decimate.SetTargetReduction(0.5)
    decimate.PreserveTopologyOn()
    decimate.Update()
    vtk_image = decimate.GetOutput()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(vtk_image)
    normals.SetFeatureAngle(60)
    normals.Update()
    vtk_image = normals.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vtk_image)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.9, 0.9, 0.9)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.Render()
    render_window_interactor.Start()


if __name__ == "__main__":
    main()