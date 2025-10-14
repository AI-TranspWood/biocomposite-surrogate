import numpy as np
import json

keyOrder = []
keyOrder.append("fiberOrientationDistribution")
keyOrder.append("fiberAspectRatio")
keyOrder.append("orientationConcentration")
keyOrder.append("microfibrilAngle")
keyOrder.append("lumenPorosity")
keyOrder.append("fiberVolumeFraction")
keyOrder.append("celluloseContent")
keyOrder.append("hemicelluloseContent")
keyOrder.append("ligninContent")
keyOrder.append("pectinContent")
keyOrder.append("extractivesContent")
keyOrder.append("ashContent")
keyOrder.append("celluloseCrystallinity")
keyOrder.append("matrixYoungsModulus")
keyOrder.append("matrixPoissonRatio")
keyOrder.append("airPorosity")
keyOrder.append("tangentialInterfaceCompliance")
keyOrder.append("longitudinalInterfaceCompliance")

unitTransform = lambda x : x

limits = []
limits.append([ 0, 1, unitTransform])
limits.append([ 1.2, 20000, np.log10])
limits.append([ 1e-10, 100, np.log10])
limits.append([ 0, 45, unitTransform])
limits.append([ 0, 0.99, unitTransform])
limits.append([ 0, 1, unitTransform])
limits.append([ 0.05, 0.9, unitTransform])
limits.append([ 0.01, 0.5, unitTransform])
limits.append([ 0.01, 0.6, unitTransform])
limits.append([ 0, 0.15, unitTransform])
limits.append([ 0, 0.15, unitTransform])
limits.append([ 0, 0.20, unitTransform])
limits.append([ 0.20, 1.00, unitTransform])
limits.append([ 0, 100, unitTransform])
limits.append([ 0, 0.5, unitTransform])
limits.append([ 0, 0.5, unitTransform])
limits.append([ 1e-10, 1e5, np.log10])
limits.append([ 1e-10, 1e5, np.log10])

def loadInput( filePath ):

    file = open( filePath )
    data = json.load( file )

    xDims = ( 1, 18, 1)

    X = np.empty( xDims )

    for ii, key_ii in enumerate( keyOrder ):

        input_ii = data[ key_ii ]
        scaledInput_ii = scale( input_ii, ii)

        X[ 0, ii, 0] = scaledInput_ii

    X[ 0, 6:11, 0] = X[ 0, 6:11, 0] / np.sum( X[ 0, 6:11, 0] )
    X = X - 0.5

    return X

def saveResult( result, filePath):
    
    with open( filePath, 'w', encoding='utf-8') as file:
        json.dump( result, file, ensure_ascii = False, indent = 2)


def scale( x, ii):

    errorMessage = "Invalid value for key: "
    errorMessage += keyOrder[ii]

    if ii == 0:
        if x == "vonMises":
            output = 0
        elif x == "vonMisesFisher":
            output = 1
        else:
            errorMessage += ". Valid values are vonMises or vonMisesFisher"
            raise ValueError( errorMessage )
    else:

        limits_ii = limits[ii]

        min = limits_ii[0]
        max = limits_ii[1]
        
        inInterval = ( min <= x ) and ( x <= max )

        if not inInterval:
            errorMessage += f". Valid values are within interval [ {min}, {max}]"
            raise ValueError( errorMessage )
        
        transform = limits_ii[2]

        min = transform( min )
        max = transform( max )
        interval = max - min

        output = transform( x ) - min
        output = output / interval

    return output        