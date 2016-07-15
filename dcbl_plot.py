#### import the simple module from the paraview
from paraview.simple import *
def main(fname, outfile):
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'NetCDF Reader'
    a0nc = NetCDFReader(FileName=[fname])
    a0nc.Dimensions = '(z, y, x)'
    a0nc.SphericalCoordinates = 1
    a0nc.VerticalScale = 1.0
    a0nc.VerticalBias = 0.0
    a0nc.ReplaceFillValueWithNan = 0
    a0nc.OutputType = 'Automatic'

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    renderView1.ViewSize = [1920,1080]

    # show data in view
    a0ncDisplay = Show(a0nc, renderView1)
    # trace defaults for the display properties.
    a0ncDisplay.CubeAxesVisibility = 0
    a0ncDisplay.Representation = 'Outline'
    a0ncDisplay.AmbientColor = [1.0, 1.0, 1.0]
    a0ncDisplay.ColorArrayName = [None, '']
    a0ncDisplay.DiffuseColor = [1.0, 1.0, 1.0]
    a0ncDisplay.LookupTable = None
    a0ncDisplay.MapScalars = 1
    a0ncDisplay.InterpolateScalarsBeforeMapping = 1
    a0ncDisplay.Opacity = 1.0
    a0ncDisplay.PointSize = 2.0
    a0ncDisplay.LineWidth = 1.0
    a0ncDisplay.Interpolation = 'Gouraud'
    a0ncDisplay.Specular = 0.0
    a0ncDisplay.SpecularColor = [1.0, 1.0, 1.0]
    a0ncDisplay.SpecularPower = 100.0
    a0ncDisplay.Ambient = 0.0
    a0ncDisplay.Diffuse = 1.0
    a0ncDisplay.EdgeColor = [0.0, 0.0, 0.5]
    a0ncDisplay.BackfaceRepresentation = 'Follow Frontface'
    a0ncDisplay.BackfaceAmbientColor = [1.0, 1.0, 1.0]
    a0ncDisplay.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
    a0ncDisplay.BackfaceOpacity = 1.0
    a0ncDisplay.Position = [0.0, 0.0, 0.0]
    a0ncDisplay.Scale = [1.0, 1.0, 1.0]
    a0ncDisplay.Orientation = [0.0, 0.0, 0.0]
    a0ncDisplay.Origin = [0.0, 0.0, 0.0]
    a0ncDisplay.Pickable = 1
    a0ncDisplay.Texture = None
    a0ncDisplay.Triangulate = 0
    a0ncDisplay.NonlinearSubdivisionLevel = 1
    a0ncDisplay.GlyphType = 'Arrow'
    a0ncDisplay.CubeAxesColor = [1.0, 1.0, 1.0]
    a0ncDisplay.CubeAxesCornerOffset = 0.0
    a0ncDisplay.CubeAxesFlyMode = 'Closest Triad'
    a0ncDisplay.CubeAxesInertia = 1
    a0ncDisplay.CubeAxesTickLocation = 'Inside'
    a0ncDisplay.CubeAxesXAxisMinorTickVisibility = 1
    a0ncDisplay.CubeAxesXAxisTickVisibility = 1
    a0ncDisplay.CubeAxesXAxisVisibility = 1
    a0ncDisplay.CubeAxesXGridLines = 0
    a0ncDisplay.CubeAxesXTitle = 'X-Axis'
    a0ncDisplay.CubeAxesUseDefaultXTitle = 1
    a0ncDisplay.CubeAxesYAxisMinorTickVisibility = 1
    a0ncDisplay.CubeAxesYAxisTickVisibility = 1
    a0ncDisplay.CubeAxesYAxisVisibility = 1
    a0ncDisplay.CubeAxesYGridLines = 0
    a0ncDisplay.CubeAxesYTitle = 'Y-Axis'
    a0ncDisplay.CubeAxesUseDefaultYTitle = 1
    a0ncDisplay.CubeAxesZAxisMinorTickVisibility = 1
    a0ncDisplay.CubeAxesZAxisTickVisibility = 1
    a0ncDisplay.CubeAxesZAxisVisibility = 1
    a0ncDisplay.CubeAxesZGridLines = 0
    a0ncDisplay.CubeAxesZTitle = 'Z-Axis'
    a0ncDisplay.CubeAxesUseDefaultZTitle = 1
    a0ncDisplay.CubeAxesGridLineLocation = 'All Faces'
    a0ncDisplay.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a0ncDisplay.CustomBoundsActive = [0, 0, 0]
    a0ncDisplay.OriginalBoundsRangeActive = [0, 0, 0]
    a0ncDisplay.CustomRange = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a0ncDisplay.CustomRangeActive = [0, 0, 0]
    a0ncDisplay.UseAxesOrigin = 0
    a0ncDisplay.AxesOrigin = [0.0, 0.0, 0.0]
    a0ncDisplay.CubeAxesXLabelFormat = '%-#6.3g'
    a0ncDisplay.CubeAxesYLabelFormat = '%-#6.3g'
    a0ncDisplay.CubeAxesZLabelFormat = '%-#6.3g'
    a0ncDisplay.StickyAxes = 0
    a0ncDisplay.CenterStickyAxes = 0
    a0ncDisplay.SelectionCellLabelBold = 0
    a0ncDisplay.SelectionCellLabelColor = [0.0, 1.0, 0.0]
    a0ncDisplay.SelectionCellLabelFontFamily = 'Arial'
    a0ncDisplay.SelectionCellLabelFontSize = 18
    a0ncDisplay.SelectionCellLabelItalic = 0
    a0ncDisplay.SelectionCellLabelJustification = 'Left'
    a0ncDisplay.SelectionCellLabelOpacity = 1.0
    a0ncDisplay.SelectionCellLabelShadow = 0
    a0ncDisplay.SelectionPointLabelBold = 0
    a0ncDisplay.SelectionPointLabelColor = [1.0, 1.0, 0.0]
    a0ncDisplay.SelectionPointLabelFontFamily = 'Arial'
    a0ncDisplay.SelectionPointLabelFontSize = 18
    a0ncDisplay.SelectionPointLabelItalic = 0
    a0ncDisplay.SelectionPointLabelJustification = 'Left'
    a0ncDisplay.SelectionPointLabelOpacity = 1.0
    a0ncDisplay.SelectionPointLabelShadow = 0
    a0ncDisplay.ScalarOpacityUnitDistance = 235.15101530718516
    a0ncDisplay.VolumeRenderingMode = 'Smart'
    a0ncDisplay.Shade = 0
    a0ncDisplay.SliceMode = 'XY Plane'
    a0ncDisplay.Slice = 15

    # init the 'Arrow' selected for 'GlyphType'
    a0ncDisplay.GlyphType.TipResolution = 6
    a0ncDisplay.GlyphType.TipRadius = 0.1
    a0ncDisplay.GlyphType.TipLength = 0.35
    a0ncDisplay.GlyphType.ShaftResolution = 6
    a0ncDisplay.GlyphType.ShaftRadius = 0.03
    a0ncDisplay.GlyphType.Invert = 0

    # reset view to fit data
    renderView1.ResetCamera()

    # find source
    a0nc_1 = FindSource('0.nc')

    # find source
    a0nc_2 = FindSource('0.nc')

    # set scalar coloring
    ColorBy(a0ncDisplay, ('POINTS', 'buoyancy_frequency'))

    # rescale color and/or opacity maps used to include current data range
    a0ncDisplay.RescaleTransferFunctionToDataRange(True)

    # change representation type
    a0ncDisplay.SetRepresentationType('Volume')

    # get color transfer function/color map for 'buoyancyfrequency'
    buoyancyfrequencyLUT = GetColorTransferFunction('buoyancyfrequency')
    buoyancyfrequencyLUT.LockDataRange = 0
    buoyancyfrequencyLUT.InterpretValuesAsCategories = 0
    buoyancyfrequencyLUT.ShowCategoricalColorsinDataRangeOnly = 0
    buoyancyfrequencyLUT.RescaleOnVisibilityChange = 0
    buoyancyfrequencyLUT.EnableOpacityMapping = 0
    buoyancyfrequencyLUT.RGBPoints = [-0.0004340280353267949, 0.231373, 0.298039, 0.752941, 0.000811992148244456, 0.865003, 0.865003, 0.865003, 0.002058012331815707, 0.705882, 0.0156863, 0.14902]
    buoyancyfrequencyLUT.UseLogScale = 0
    buoyancyfrequencyLUT.ColorSpace = 'Diverging'
    buoyancyfrequencyLUT.UseBelowRangeColor = 0
    buoyancyfrequencyLUT.BelowRangeColor = [0.0, 0.0, 0.0]
    buoyancyfrequencyLUT.UseAboveRangeColor = 0
    buoyancyfrequencyLUT.AboveRangeColor = [1.0, 1.0, 1.0]
    buoyancyfrequencyLUT.NanColor = [1.0, 1.0, 0.0]
    buoyancyfrequencyLUT.Discretize = 1
    buoyancyfrequencyLUT.NumberOfTableValues = 256
    buoyancyfrequencyLUT.ScalarRangeInitialized = 1.0
    buoyancyfrequencyLUT.HSVWrap = 0
    buoyancyfrequencyLUT.VectorComponent = 0
    buoyancyfrequencyLUT.VectorMode = 'Magnitude'
    buoyancyfrequencyLUT.AllowDuplicateScalars = 1
    buoyancyfrequencyLUT.Annotations = []
    buoyancyfrequencyLUT.ActiveAnnotatedValues = []
    buoyancyfrequencyLUT.IndexedColors = []

    # get opacity transfer function/opacity map for 'buoyancyfrequency'
    buoyancyfrequencyPWF = GetOpacityTransferFunction('buoyancyfrequency')
    buoyancyfrequencyPWF.Points = [-0.0004340280353267949, 0.0, 0.5, 0.0, 0.002058012331815707, 1.0, 0.5, 0.0]
    buoyancyfrequencyPWF.AllowDuplicateScalars = 1
    buoyancyfrequencyPWF.ScalarRangeInitialized = 1

    # set scalar coloring
    ColorBy(a0ncDisplay, ('POINTS', 'w'))

    # rescale color and/or opacity maps used to include current data range
    a0ncDisplay.RescaleTransferFunctionToDataRange(True)

    # show color bar/color legend
    a0ncDisplay.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'w'
    wLUT = GetColorTransferFunction('w')
    wLUT.LockDataRange = 0
    wLUT.InterpretValuesAsCategories = 0
    wLUT.ShowCategoricalColorsinDataRangeOnly = 0
    wLUT.RescaleOnVisibilityChange = 0
    wLUT.EnableOpacityMapping = 0
    wLUT.RGBPoints = [-2.7832190021186833, 0.231373, 0.298039, 0.752941, 0.922164977609047, 0.865003, 0.865003, 0.865003, 4.627548957336778, 0.705882, 0.0156863, 0.14902]
    wLUT.UseLogScale = 0
    wLUT.ColorSpace = 'Diverging'
    wLUT.UseBelowRangeColor = 0
    wLUT.BelowRangeColor = [0.0, 0.0, 0.0]
    wLUT.UseAboveRangeColor = 0
    wLUT.AboveRangeColor = [1.0, 1.0, 1.0]
    wLUT.NanColor = [1.0, 1.0, 0.0]
    wLUT.Discretize = 1
    wLUT.NumberOfTableValues = 256
    wLUT.ScalarRangeInitialized = 1.0
    wLUT.HSVWrap = 0
    wLUT.VectorComponent = 0
    wLUT.VectorMode = 'Magnitude'
    wLUT.AllowDuplicateScalars = 1
    wLUT.Annotations = []
    wLUT.ActiveAnnotatedValues = []
    wLUT.IndexedColors = []

    # get opacity transfer function/opacity map for 'w'
    wPWF = GetOpacityTransferFunction('w')
    wPWF.Points = [-2.7832190021186833, 0.0, 0.5, 0.0, 1.2417986914323156, 0.0, 0.5, 0.0, 1.833713132357674, 0.737500011920929, 0.5, 0.0, 4.627548957336778, 1.0, 0.5, 0.0]
    wPWF.AllowDuplicateScalars = 1
    wPWF.ScalarRangeInitialized = 1

    # create a new 'NetCDF Reader'
    a0nc_3 = NetCDFReader(FileName=[fname])
    a0nc_3.Dimensions = '(z, y, x)'
    a0nc_3.SphericalCoordinates = 1
    a0nc_3.VerticalScale = 1.0
    a0nc_3.VerticalBias = 0.0
    a0nc_3.ReplaceFillValueWithNan = 0
    a0nc_3.OutputType = 'Automatic'

    # show data in view
    a0nc_3Display = Show(a0nc_3, renderView1)
    # trace defaults for the display properties.
    a0nc_3Display.CubeAxesVisibility = 0
    a0nc_3Display.Representation = 'Outline'
    a0nc_3Display.AmbientColor = [1.0, 1.0, 1.0]
    a0nc_3Display.ColorArrayName = [None, '']
    a0nc_3Display.DiffuseColor = [1.0, 1.0, 1.0]
    a0nc_3Display.LookupTable = None
    a0nc_3Display.MapScalars = 1
    a0nc_3Display.InterpolateScalarsBeforeMapping = 1
    a0nc_3Display.Opacity = 1.0
    a0nc_3Display.PointSize = 2.0
    a0nc_3Display.LineWidth = 1.0
    a0nc_3Display.Interpolation = 'Gouraud'
    a0nc_3Display.Specular = 0.0
    a0nc_3Display.SpecularColor = [1.0, 1.0, 1.0]
    a0nc_3Display.SpecularPower = 100.0
    a0nc_3Display.Ambient = 0.0
    a0nc_3Display.Diffuse = 1.0
    a0nc_3Display.EdgeColor = [0.0, 0.0, 0.5]
    a0nc_3Display.BackfaceRepresentation = 'Follow Frontface'
    a0nc_3Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
    a0nc_3Display.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
    a0nc_3Display.BackfaceOpacity = 1.0
    a0nc_3Display.Position = [0.0, 0.0, 0.0]
    a0nc_3Display.Scale = [1.0, 1.0, 1.0]
    a0nc_3Display.Orientation = [0.0, 0.0, 0.0]
    a0nc_3Display.Origin = [0.0, 0.0, 0.0]
    a0nc_3Display.Pickable = 1
    a0nc_3Display.Texture = None
    a0nc_3Display.Triangulate = 0
    a0nc_3Display.NonlinearSubdivisionLevel = 1
    a0nc_3Display.GlyphType = 'Arrow'
    a0nc_3Display.CubeAxesColor = [1.0, 1.0, 1.0]
    a0nc_3Display.CubeAxesCornerOffset = 0.0
    a0nc_3Display.CubeAxesFlyMode = 'Closest Triad'
    a0nc_3Display.CubeAxesInertia = 1
    a0nc_3Display.CubeAxesTickLocation = 'Inside'
    a0nc_3Display.CubeAxesXAxisMinorTickVisibility = 1
    a0nc_3Display.CubeAxesXAxisTickVisibility = 1
    a0nc_3Display.CubeAxesXAxisVisibility = 1
    a0nc_3Display.CubeAxesXGridLines = 0
    a0nc_3Display.CubeAxesXTitle = 'X-Axis'
    a0nc_3Display.CubeAxesUseDefaultXTitle = 1
    a0nc_3Display.CubeAxesYAxisMinorTickVisibility = 1
    a0nc_3Display.CubeAxesYAxisTickVisibility = 1
    a0nc_3Display.CubeAxesYAxisVisibility = 1
    a0nc_3Display.CubeAxesYGridLines = 0
    a0nc_3Display.CubeAxesYTitle = 'Y-Axis'
    a0nc_3Display.CubeAxesUseDefaultYTitle = 1
    a0nc_3Display.CubeAxesZAxisMinorTickVisibility = 1
    a0nc_3Display.CubeAxesZAxisTickVisibility = 1
    a0nc_3Display.CubeAxesZAxisVisibility = 1
    a0nc_3Display.CubeAxesZGridLines = 0
    a0nc_3Display.CubeAxesZTitle = 'Z-Axis'
    a0nc_3Display.CubeAxesUseDefaultZTitle = 1
    a0nc_3Display.CubeAxesGridLineLocation = 'All Faces'
    a0nc_3Display.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a0nc_3Display.CustomBoundsActive = [0, 0, 0]
    a0nc_3Display.OriginalBoundsRangeActive = [0, 0, 0]
    a0nc_3Display.CustomRange = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a0nc_3Display.CustomRangeActive = [0, 0, 0]
    a0nc_3Display.UseAxesOrigin = 0
    a0nc_3Display.AxesOrigin = [0.0, 0.0, 0.0]
    a0nc_3Display.CubeAxesXLabelFormat = '%-#6.3g'
    a0nc_3Display.CubeAxesYLabelFormat = '%-#6.3g'
    a0nc_3Display.CubeAxesZLabelFormat = '%-#6.3g'
    a0nc_3Display.StickyAxes = 0
    a0nc_3Display.CenterStickyAxes = 0
    a0nc_3Display.SelectionCellLabelBold = 0
    a0nc_3Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
    a0nc_3Display.SelectionCellLabelFontFamily = 'Arial'
    a0nc_3Display.SelectionCellLabelFontSize = 18
    a0nc_3Display.SelectionCellLabelItalic = 0
    a0nc_3Display.SelectionCellLabelJustification = 'Left'
    a0nc_3Display.SelectionCellLabelOpacity = 1.0
    a0nc_3Display.SelectionCellLabelShadow = 0
    a0nc_3Display.SelectionPointLabelBold = 0
    a0nc_3Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
    a0nc_3Display.SelectionPointLabelFontFamily = 'Arial'
    a0nc_3Display.SelectionPointLabelFontSize = 18
    a0nc_3Display.SelectionPointLabelItalic = 0
    a0nc_3Display.SelectionPointLabelJustification = 'Left'
    a0nc_3Display.SelectionPointLabelOpacity = 1.0
    a0nc_3Display.SelectionPointLabelShadow = 0
    a0nc_3Display.ScalarOpacityUnitDistance = 235.15101530718516
    a0nc_3Display.VolumeRenderingMode = 'Smart'
    a0nc_3Display.Shade = 0
    a0nc_3Display.SliceMode = 'XY Plane'
    a0nc_3Display.Slice = 15

    # init the 'Arrow' selected for 'GlyphType'
    a0nc_3Display.GlyphType.TipResolution = 6
    a0nc_3Display.GlyphType.TipRadius = 0.1
    a0nc_3Display.GlyphType.TipLength = 0.35
    a0nc_3Display.GlyphType.ShaftResolution = 6
    a0nc_3Display.GlyphType.ShaftRadius = 0.03
    a0nc_3Display.GlyphType.Invert = 0

    # set scalar coloring
    ColorBy(a0nc_3Display, ('POINTS', 'buoyancy_frequency'))

    # rescale color and/or opacity maps used to include current data range
    a0nc_3Display.RescaleTransferFunctionToDataRange(True)

    # change representation type
    a0nc_3Display.SetRepresentationType('Slice')

    # Properties modified on a0nc_3Display
    a0nc_3Display.Slice = 14

    # Properties modified on a0nc_3Display
    a0nc_3Display.Slice = 12

    # Properties modified on a0nc_3Display
    a0nc_3Display.Slice = 4

    # Properties modified on a0nc_3Display
    a0nc_3Display.Slice = 0

    # rescale color and/or opacity maps used to exactly fit the current data range
    a0nc_3Display.RescaleTransferFunctionToDataRange(False)

    # set scalar coloring
    ColorBy(a0nc_3Display, ('POINTS', 's'))

    # rescale color and/or opacity maps used to include current data range
    a0nc_3Display.RescaleTransferFunctionToDataRange(True)

    # show color bar/color legend
    a0nc_3Display.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 's'
    sLUT = GetColorTransferFunction('s')
    sLUT.LockDataRange = 0
    sLUT.InterpretValuesAsCategories = 0
    sLUT.ShowCategoricalColorsinDataRangeOnly = 0
    sLUT.RescaleOnVisibilityChange = 0
    sLUT.EnableOpacityMapping = 0
    sLUT.RGBPoints = [6872.497301437454, 0.0, 0.0, 0.0, 6874.656767490042, 0.239216, 0.027451, 0.415686, 6876.8162335426305, 0.0901961, 0.254902, 0.556863, 6878.97571680349, 0.0941176, 0.352941, 0.54902, 6881.135182856078, 0.105882, 0.435294, 0.533333, 6883.294648908667, 0.12549, 0.52549, 0.501961, 6885.454114961255, 0.156863, 0.596078, 0.443137, 6887.613581013843, 0.196078, 0.65098, 0.380392, 6889.773056875147, 0.282353, 0.717647, 0.301961, 6891.932530327292, 0.466667, 0.772549, 0.27451, 6894.09199637988, 0.678431, 0.784314, 0.309804, 6896.2514624324685, 0.901961, 0.756863, 0.376471, 6898.4109284850565, 0.992157, 0.705882, 0.521569, 6900.570411745916, 1.0, 0.721569, 0.701961, 6902.729877798505, 1.0, 0.784314, 0.784314, 6904.889343851093, 1.0, 0.866667, 0.866667, 6906.913845426428, 1.0, 1.0, 1.0]
    sLUT.UseLogScale = 0
    sLUT.ColorSpace = 'Lab'
    sLUT.UseBelowRangeColor = 0
    sLUT.BelowRangeColor = [0.0, 0.0, 0.0]
    sLUT.UseAboveRangeColor = 0
    sLUT.AboveRangeColor = [1.0, 1.0, 1.0]
    sLUT.NanColor = [1.0, 1.0, 0.0]
    sLUT.Discretize = 1
    sLUT.NumberOfTableValues = 256
    sLUT.ScalarRangeInitialized = 1.0
    sLUT.HSVWrap = 0
    sLUT.VectorComponent = 0
    sLUT.VectorMode = 'Magnitude'
    sLUT.AllowDuplicateScalars = 1
    sLUT.Annotations = []
    sLUT.ActiveAnnotatedValues = []
    sLUT.IndexedColors = []

    # get opacity transfer function/opacity map for 's'
    sPWF = GetOpacityTransferFunction('s')
    sPWF.Points = [6872.497301437454, 0.0, 0.5, 0.0, 6880.4140625, 0.5562500357627869, 0.5, 0.0, 6888.111328125, 0.59375, 0.5, 0.0, 6890.09033203125, 1.0, 0.5, 0.0, 6906.913845426428, 1.0, 0.5, 0.0]
    sPWF.AllowDuplicateScalars = 1
    sPWF.ScalarRangeInitialized = 1

    # Properties modified on sPWF
    sPWF.Points = [6872.497301437454, 0.0, 0.5, 0.0, 6880.4140625, 0.5562500357627869, 0.5, 0.0, 6885.66357421875, 0.75, 0.5, 0.0, 6890.09033203125, 1.0, 0.5, 0.0, 6906.913845426428, 1.0, 0.5, 0.0]

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    sLUT.ApplyPreset('Cool to Warm', True)

    # Properties modified on sPWF
    sPWF.Points = [6872.497301437454, 0.0, 0.5, 0.0, 6880.4140625, 0.5562500357627869, 0.5, 0.0, 6885.66357421875, 0.75, 0.5, 0.0, 6896.90771484375, 0.612500011920929, 0.5, 0.0, 6906.913845426428, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.497301437454, 0.0, 0.5, 0.0, 6880.4140625, 0.5562500357627869, 0.5, 0.0, 6885.66357421875, 0.4625000059604645, 0.5, 0.0, 6885.66357421875, 0.75, 0.5, 0.0, 6896.90771484375, 0.612500011920929, 0.5, 0.0, 6906.913845426428, 1.0, 0.5, 0.0]

    # Rescale transfer function
    sLUT.RescaleTransferFunction(6872.5, 6880.91)

    # Rescale transfer function
    sPWF.RescaleTransferFunction(6872.5, 6880.91)

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.434533593999, 0.5562500357627869, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.55712890625, 0.5250000357627869, 0.5, 0.0, 6878.464909690314, 0.612500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.434533593999, 0.5562500357627869, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.7451171875, 0.4937500059604645, 0.5, 0.0, 6878.464909690314, 0.612500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.434533593999, 0.5562500357627869, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.8525390625, 0.5, 0.5, 0.0, 6878.464909690314, 0.612500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.434533593999, 0.5562500357627869, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.8525390625, 0.5, 0.5, 0.0, 6878.464909690314, 0.612500011920929, 0.5, 0.0, 6878.70654296875, 0.625, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.434533593999, 0.5562500357627869, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.8525390625, 0.5, 0.5, 0.0, 6878.464909690314, 0.612500011920929, 0.5, 0.0, 6879.13671875, 0.856249988079071, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.434533593999, 0.5562500357627869, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.8525390625, 0.5, 0.5, 0.0, 6878.464909690314, 0.612500011920929, 0.5, 0.0, 6879.21728515625, 0.862500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.434533593999, 0.5562500357627869, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.8525390625, 0.5, 0.5, 0.0, 6877.90087890625, 0.768750011920929, 0.5, 0.0, 6879.21728515625, 0.862500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.434533593999, 0.5562500357627869, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.8525390625, 0.5, 0.5, 0.0, 6877.1845703125, 0.706250011920929, 0.5, 0.0, 6877.90087890625, 0.768750011920929, 0.5, 0.0, 6879.21728515625, 0.862500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.434533593999, 0.5562500357627869, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.1689453125, 0.5875000357627869, 0.5, 0.0, 6877.1845703125, 0.706250011920929, 0.5, 0.0, 6877.90087890625, 0.768750011920929, 0.5, 0.0, 6879.21728515625, 0.862500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.4189453125, 0.375, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.1689453125, 0.5875000357627869, 0.5, 0.0, 6877.1845703125, 0.706250011920929, 0.5, 0.0, 6877.90087890625, 0.768750011920929, 0.5, 0.0, 6879.21728515625, 0.862500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6874.30615234375, 0.29375001788139343, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.1689453125, 0.5875000357627869, 0.5, 0.0, 6877.1845703125, 0.706250011920929, 0.5, 0.0, 6877.90087890625, 0.768750011920929, 0.5, 0.0, 6879.21728515625, 0.862500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6873.939453125, 0.23125000298023224, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.1689453125, 0.5875000357627869, 0.5, 0.0, 6877.1845703125, 0.706250011920929, 0.5, 0.0, 6877.90087890625, 0.768750011920929, 0.5, 0.0, 6879.21728515625, 0.862500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Properties modified on sPWF
    sPWF.Points = [6872.5, 0.0, 0.5, 0.0, 6873.74169921875, 0.22500000894069672, 0.5, 0.0, 6875.717300206731, 0.4625000059604645, 0.5, 0.0, 6876.1689453125, 0.5875000357627869, 0.5, 0.0, 6877.1845703125, 0.706250011920929, 0.5, 0.0, 6877.90087890625, 0.768750011920929, 0.5, 0.0, 6879.21728515625, 0.862500011920929, 0.5, 0.0, 6880.91, 1.0, 0.5, 0.0]

    # Rescale transfer function
    sLUT.RescaleTransferFunction(6872.5, 6890.91)

    # Rescale transfer function
    sPWF.RescaleTransferFunction(6872.5, 6890.91)

    # current camera placement for renderView1
# current camera placement for renderView1
    renderView1.CameraPosition = [-5033.443897132494, 9539.33223527591, 3170.7253473326746]
    renderView1.CameraFocalPoint = [2459.599516453724, 2504.042384292785, 868.2432468785019]
    renderView1.CameraViewUp = [0.17736664543244707, -0.13034263521218856, 0.9754752024187919]
    renderView1.CameraParallelScale = 3644.840737261369

    # save screenshot
    #SaveScreenshot(outfile, magnification=1, quality=100, view=renderView1)
    Render()
    WriteImage(outfile)


    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("fname")
    parser.add_argument("outfile")
    args = parser.parse_args()

    main(args.fname,  args.outfile)