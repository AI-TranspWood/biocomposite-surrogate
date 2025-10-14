import os

import h5py
import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import tfkan as tft


def createArchitecture( inputSize, outputSize, layerSize = 1):

    tf.keras.backend.clear_session()

    inputShape = ( inputSize, 1)

    elasticModuli = lambda input: tfp.distributions.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:2] ) - 10, scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )
    poissonRatios = lambda input: tfp.distributions.Normal( loc   = 6 * tf.nn.sigmoid( input[..., 0:2] ) - 3, scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )
    elasticModulus = lambda input: tfp.distributions.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:1] ) - 10, scale = 10 * tf.nn.sigmoid( input[..., 1:2] ) + 1e-6 )

    elasticModuliLayer = tfp.layers.DistributionLambda( elasticModuli )
    poissonRatioLayer = tfp.layers.DistributionLambda( poissonRatios )
    elasticModulusLayer = tfp.layers.DistributionLambda( elasticModulus )

    inputs = keras.layers.Input( shape = inputShape )

    xBayesian = tfp.layers.DenseFlipout( layerSize )( 0 * inputs )
    x = tft.layers.DenseKAN( layerSize )( inputs )

    x = keras.layers.concatenate([ xBayesian / np.sqrt( layerSize ), x])
    x = keras.layers.Flatten()( x )
    x = tft.layers.DenseKAN( 5 * layerSize )( x )
    x = keras.layers.Flatten()( x )
    x = tft.layers.DenseKAN( 5 * layerSize )( x )
    x = keras.layers.Flatten()( x )
    x = tft.layers.DenseKAN( outputSize + outputSize )( x )

    E = elasticModuliLayer( x[ ..., 0:4] )
    nu = poissonRatioLayer( x[ ..., 4:8] )
    mu = elasticModulusLayer( x[ ..., 8:10] )

    model = keras.Model( inputs = inputs, outputs = [ E, nu, mu])
    model = setWeights( model )

    return model


def setWeights( model ):

    curr_dir = os.path.dirname( os.path.abspath( __file__ ) )
    file = h5py.File( os.path.join( curr_dir, "weights/weights.h5"), 'r')

    keys = file.keys()
    nKeys = len( keys )
    weights = []

    for ii in range( nKeys ):

        key_ii = 'weight' + str(ii)
        weights_ii = file[ key_ii ][:]

        weights.append( weights_ii )
    
    model.set_weights(weights)
    return model


def computeQuantiles( outputDistribution, quantiles, log = True):
    
    if log: transfrom = np.exp
    else: transfrom = lambda x : x

    lowerBoundPercentage = quantiles[0]
    upperBoundPercentage = quantiles[1]

    median = outputDistribution.quantile( 0.50 ).numpy().flatten()
    lowerBound = outputDistribution.quantile( lowerBoundPercentage ).numpy().flatten()
    upperBound = outputDistribution.quantile( upperBoundPercentage ).numpy().flatten()

    median = transfrom( median )
    lowerBound = transfrom( lowerBound )
    upperBound = transfrom( upperBound )

    result = {}
    result["median"] = median.tolist()[0]
    result["lowerBound"] = lowerBound.tolist()[0]
    result["upperBound"] = upperBound.tolist()[0]

    return result

def computeEstimate( input, model, quantiles):

    stiffnessEstimate = model( input )

    logE_T = tfp.distributions.Normal( loc = stiffnessEstimate[0].loc[ 0, 0], scale = stiffnessEstimate[0].scale[ 0, 0] )
    logE_L = tfp.distributions.Normal( loc = stiffnessEstimate[0].loc[ 0, 1], scale = stiffnessEstimate[0].scale[ 0, 1] )
    logMu_LT = tfp.distributions.Normal( loc = stiffnessEstimate[2].loc[ 0, 0], scale = stiffnessEstimate[0].scale[ 0, 0] )

    nu_T = tfp.distributions.Normal( loc = stiffnessEstimate[1].loc[ 0, 0], scale = stiffnessEstimate[0].scale[ 0, 0] )
    nu_LT = tfp.distributions.Normal( loc = stiffnessEstimate[1].loc[ 0, 1], scale = stiffnessEstimate[1].scale[ 0, 1] )

    estimateResult = {}
    estimateResult["E_T"] = computeQuantiles( logE_T, quantiles = quantiles)
    estimateResult["E_L"] = computeQuantiles( logE_L, quantiles = quantiles)
    estimateResult["nu_T"] = computeQuantiles( nu_T, quantiles = quantiles, log = False)
    estimateResult["nu_LT"] = computeQuantiles( nu_LT, quantiles = quantiles, log = False)
    estimateResult["mu_LT"] = computeQuantiles( logMu_LT, quantiles = quantiles)
    estimateResult["quantiles"] = quantiles
    
    return estimateResult