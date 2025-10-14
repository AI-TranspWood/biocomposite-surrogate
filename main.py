import sys
import numpy as np

import include.surrogate as surrogate
import include.dataLoaders as dl

inputFilePath = sys.argv[1]
outputFilePath = sys.argv[2]

lowerBoundPercentage = float( sys.argv[3] )
upperBoundPercentage = float( sys.argv[4] )
quantiles = [ lowerBoundPercentage, upperBoundPercentage]

input = dl.loadInput( inputFilePath )

model = surrogate.createArchitecture( inputSize = 18, outputSize = 5, layerSize = 24)
estimate = surrogate.computeEstimate( input, model, quantiles = quantiles)

dl.saveResult( estimate, outputFilePath)