/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import com.beust.klaxon.Klaxon
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.activation.*
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.*
import org.jetbrains.kotlinx.dl.api.core.layer.core.ActivationLayer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.merge.*
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.*
import org.jetbrains.kotlinx.dl.api.core.layer.regularization.Dropout
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Cropping2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.RepeatVector
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Reshape
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.ZeroPadding2D
import org.jetbrains.kotlinx.dl.api.core.regularizer.L1
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2L1
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.inference.keras.config.*
import java.io.File

/**
 * Loads a [Sequential] model from json file with model configuration.
 *
 * @param [configuration] File containing model configuration.
 * @return Non-compiled and non-trained Sequential model.
 */
internal fun loadSequentialModelConfiguration(
    configuration: File
): Sequential {
    val sequentialConfig = loadSerializedModel(configuration)
    return deserializeSequentialModel(sequentialConfig)
}

internal fun deserializeSequentialModel(sequentialConfig: KerasModel?): Sequential {
    val pair = loadSequentialModelLayers(sequentialConfig)
    val input: Input = pair.first
    val layers = pair.second

    return Sequential.of(input, *layers.toList().toTypedArray())
}

/**
 * Loads a [Sequential] model layers from json file with model configuration.
 *
 * NOTE: This method is useful in transfer learning, when you need to manipulate on layers before building the Sequential model.
 *
 * @param config Model configuration.
 * @return Pair of <input layer; list of layers>.
 */
internal fun loadSequentialModelLayers(config: KerasModel?): Pair<Input, MutableList<Layer>> {
    val layers = mutableListOf<Layer>()

    (config as KerasModel).config!!.layers!!.forEach {
        run {
            if (!it.class_name.equals("InputLayer")) {
                val layer = convertToLayer(it)
                layers.add(layer)
            }
        }
    }

    val input: Input

    val firstLayer = config.config!!.layers!!.first()
    val inputLayerName =
        if (firstLayer.class_name.equals("InputLayer")) firstLayer.config!!.name ?: "input" else "input"
    val batchInputShape = config.config.layers!!.first().config!!.batch_input_shape

    // TODO: write more universal code here
    when (batchInputShape!!.size) {
        3 -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                name = inputLayerName
            )
        }
        4 -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                batchInputShape[3]?.toLong()!!,
                name = inputLayerName
            )
        }
        else -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                batchInputShape[3]?.toLong()!!,
                name = inputLayerName
            )
        }
    }

    return Pair(input, layers)
}

private fun convertToLayer(
    kerasLayer: KerasLayer
): Layer {
    return when (kerasLayer.class_name) {
        // Core layers
        LAYER_ACTIVATION -> createActivationLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_DENSE -> createDenseLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        // Convolution layers
        LAYER_CONV1D -> createConv1DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_CONV2D -> createConv2DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_CONV3D -> createConv3DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_DEPTHWISE_CONV2D -> createDepthwiseConv2DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_SEPARABLE_CONV2D -> createSeparableConv2DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        // Pooling layers
        LAYER_MAX_POOL_1D -> createMaxPool1DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_MAX_POOL_2D -> createMaxPool2DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_MAX_POOL_3D -> createMaxPool3DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_AVG_POOL_1D -> createAvgPool1DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_AVG_POOL_2D -> createAvgPool2DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_AVG_POOL_3D -> createAvgPool3DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_GLOBAL_MAX_POOL_1D -> createGlobalMaxPool1DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_GLOBAL_MAX_POOL_2D -> createGlobalMaxPool2DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_GLOBAL_MAX_POOL_3D -> createGlobalMaxPool3DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_GLOBAL_AVG_POOL_1D -> createGlobalAvgPool1DLayer(kerasLayer.config!!.name!!)
        LAYER_GLOBAL_AVG_POOL_2D -> createGlobalAvgPool2DLayer(kerasLayer.config!!.name!!)
        LAYER_GLOBAL_AVG_POOL_3D -> createGlobalAvgPool3DLayer(kerasLayer.config!!.name!!)
        // Recurrent layers
        // Normalization layers
        LAYER_BATCH_NORM -> createBatchNormLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        // Regularization layers
        LAYER_DROPOUT -> createDropoutLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        // Attention layers
        // Reshaping layers
        LAYER_FLATTEN -> createFlattenLayer(kerasLayer.config!!.name!!)
        LAYER_REPEAT_VECTOR -> createRepeatVectorLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_RESHAPE -> createReshapeLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_CROPPING_2D -> createCropping2DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ZERO_PADDING_2D -> createZeroPadding2DLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        // Merging layers
        LAYER_ADD -> createAddLayer(kerasLayer.config!!.name!!)
        LAYER_AVERAGE -> createAverageLayer(kerasLayer.config!!.name!!)
        LAYER_SUBTRACT -> createSubtractLayer(kerasLayer.config!!.name!!)
        LAYER_MAXIMUM -> createMaximumLayer(kerasLayer.config!!.name!!)
        LAYER_MINIMUM -> createMinimumLayer(kerasLayer.config!!.name!!)
        LAYER_MULTIPLY -> createMultiplyLayer(kerasLayer.config!!.name!!)
        LAYER_CONCATENATE -> createConcatenateLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        // Locally-connected layers
        // Activation layers
        LAYER_RELU -> createReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ELU -> createELULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_PRELU -> createPReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_LEAKY_RELU -> createLeakyReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_THRESHOLDED_RELU -> createThresholdedReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_SOFTMAX -> createSoftmaxLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        else -> throw IllegalStateException("${kerasLayer.class_name} is not supported yet!")
    }.apply {
        isTrainable = kerasLayer.config.trainable?:isTrainable
    }
}


/**
 * Loads a [Sequential] model from json file with model configuration.
 *
 * @param [configuration] File containing model configuration.
 * @return Non-compiled and non-trained Sequential model.
 */
internal fun loadFunctionalModelConfiguration(
    configuration: File
): Functional {
    val functionalConfig = loadSerializedModel(configuration)
    return deserializeFunctionalModel(functionalConfig)
}

internal fun deserializeFunctionalModel(functionalConfig: KerasModel?) =
    Functional.of(loadFunctionalModelLayers(functionalConfig).toList())

/**
 * Loads a [Functional] model layers from json file with model configuration.
 *
 * NOTE: This method is useful in transfer learning, when you need to manipulate on layers before building the Functional model.
 *
 * @param config Model configuration.
 * @return Pair of <input layer; list of layers>.
 */
internal fun loadFunctionalModelLayers(config: KerasModel?): MutableList<Layer> {
    val layers = mutableListOf<Layer>()
    val layersByNames = mutableMapOf<String, Layer>()

    val input: Input

    val firstLayer = (config as KerasModel).config!!.layers!!.first()
    val batchInputShape =
        firstLayer.config!!.batch_input_shape
    val inputLayerName =
        if (firstLayer.class_name.equals("InputLayer")) firstLayer.config.name ?: "input" else "input"

    // TODO: write more universal code here
    val size = batchInputShape!!.size
    when (size) {
        3 -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                name = inputLayerName
            )
        }
        4 -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                batchInputShape[3]?.toLong()!!,
                name = inputLayerName
            )
        }
        else -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                batchInputShape[3]?.toLong()!!,
                name = inputLayerName
            )
        }
    }

    layers.add(input)
    layersByNames[input.name] = input

    config.config!!.layers!!.forEach {
        run {
            if (!it.class_name.equals("InputLayer")) {
                val layer = convertToLayer(it, layersByNames)
                layers.add(layer)
                layersByNames[layer.name] = layer
            }
        }
    }

    return layers
}

internal fun loadSerializedModel(jsonConfigFile: File) = try {
    val jsonString = jsonConfigFile.readText(Charsets.UTF_8)
    Klaxon()
        .converter(PaddingConverter())
        .parse<KerasModel>(jsonString)
} catch (e: Exception) {
    e.printStackTrace()
    try {
        Klaxon()
            .converter(PaddingConverter())
            .parse<KerasModel>(jsonConfigFile)
    } catch (e: Exception) {
        e.printStackTrace()
        throw IllegalArgumentException("JSON file: ${jsonConfigFile.name} contains invalid JSON. The model configuration could not be loaded from this file.")
    }
}

private fun convertToLayer(
    kerasLayer: KerasLayer,
    layersByName: MutableMap<String, Layer>
): Layer {
    val layer = convertToLayer(kerasLayer)
    val inboundLayers = mutableListOf<Layer>()
    if (kerasLayer.class_name != LAYER_INPUT) {
        val inboundNodes = kerasLayer.inbound_nodes!! as List<List<List<Any>>>
        inboundNodes[0].forEach { inboundNode ->
            check(inboundNode.isNotEmpty()) { "This .json config is incorrect and could not be parsed! The list of inbound nodes for layer ${layer.name} could not be empty on this level!" }
            layersByName[inboundNode[0] as String]?.let { inboundLayers.add(it) }

        }
        layer.inboundLayers = inboundLayers
    }


    return layer
}

private fun convertToRegularizer(regularizer: KerasRegularizer?): Regularizer? {
    return if (regularizer != null) {
        val l1 = regularizer.config!!.l1
        val l2 = regularizer.config!!.l2
        if (l1 != 0.0 && l2 != 0.0) {
            L2L1(l1!!.toFloat(), l2!!.toFloat())
        } else if (l1 == 0.0 && l2 != 0.0) {
            L2(l2!!.toFloat())
        } else if (l1 != 0.0 && l2 == 0.0) {
            L1(l1!!.toFloat())
        } else {
            null
        }
    } else {
        null
    }
}

private fun convertToInitializer(initializer: KerasInitializer): Initializer {
    val seed = if (initializer.config!!.seed != null) {
        initializer.config.seed!!.toLong()
    } else 12L

    return when (initializer.class_name!!) {
        INITIALIZER_GLOROT_UNIFORM -> GlorotUniform(seed = seed)
        INITIALIZER_GLOROT_NORMAL -> GlorotNormal(seed = seed)
        INITIALIZER_HE_NORMAL -> HeNormal(seed = seed)
        INITIALIZER_HE_UNIFORM -> HeUniform(seed = seed)
        INITIALIZER_LECUN_NORMAL -> LeCunNormal(seed = seed)
        INITIALIZER_LECUN_UNIFORM -> LeCunUniform(seed = seed)
        INITIALIZER_ZEROS -> RandomUniform(
            seed = seed,
            minVal = 0.0f,
            maxVal = 0.0f
        ) // instead of real initializers, because it doesn't influence on nothing
        INITIALIZER_CONSTANT -> RandomUniform(
            seed = seed,
            minVal = 0.0f,
            maxVal = 0.0f
        ) // instead of real initializers, because it doesn't influence on nothing
        INITIALIZER_ONES -> RandomUniform(
            seed = seed,
            minVal = 1.0f,
            maxVal = 1.0f
        ) // instead of real initializers, because it doesn't influence on nothing*/
        INITIALIZER_RANDOM_NORMAL -> RandomNormal(
            seed = seed,
            mean = initializer.config.mean!!.toFloat(),
            stdev = initializer.config.stddev!!.toFloat()
        )
        INITIALIZER_RANDOM_UNIFORM -> RandomUniform(
            seed = seed,
            minVal = initializer.config.minval!!.toFloat(),
            maxVal = initializer.config.maxval!!.toFloat()
        )
        INITIALIZER_TRUNCATED_NORMAL -> TruncatedNormal(seed = seed)
        INITIALIZER_VARIANCE_SCALING -> convertVarianceScalingInitializer(initializer)
        INITIALIZER_ORTHOGONAL -> Orthogonal(seed = seed, gain = initializer.config.gain!!.toFloat())
        /*INITIALIZER_CONSTANT -> Constant(initializer.config.value!!.toFloat())*/
        INITIALIZER_IDENTITY -> Identity(initializer.config.gain?.toFloat() ?: 1f)
        else -> throw IllegalStateException("${initializer.class_name} is not supported yet!")
    }
}

private fun convertVarianceScalingInitializer(initializer: KerasInitializer): Initializer {
    val seed = if (initializer.config!!.seed != null) {
        initializer.config.seed!!.toLong()
    } else 12L

    val config = initializer.config
    val scale = config.scale!!
    val mode: Mode = convertMode(config.mode!!)
    val distribution: Distribution = convertDistribution(config.distribution!!)
    return if (scale == 2.0 && mode == Mode.FAN_IN) {
        when (distribution) {
            Distribution.UNIFORM -> HeUniform(seed)
            Distribution.TRUNCATED_NORMAL -> {
                HeNormal(seed)
            }
            else -> VarianceScaling(scale, mode, distribution, seed)
        }
    } else {
        when (mode) {
            Mode.FAN_IN -> {
                when (distribution) {
                    Distribution.UNIFORM -> LeCunUniform(seed)
                    Distribution.TRUNCATED_NORMAL -> {
                        LeCunNormal(seed)
                    }
                    else -> VarianceScaling(scale, mode, distribution, seed)
                }
            }
            Mode.FAN_AVG -> {
                when (distribution) {
                    Distribution.UNIFORM -> GlorotUniform(seed)
                    Distribution.TRUNCATED_NORMAL -> {
                        GlorotNormal(seed)
                    }
                    else -> VarianceScaling(scale, mode, distribution, seed)
                }
            }
            else -> VarianceScaling(scale, mode, distribution, seed)
        }
    }
}

private fun convertDistribution(distribution: String): Distribution {
    return when (distribution) {
        "truncated_normal" -> Distribution.TRUNCATED_NORMAL
        "uniform" -> Distribution.UNIFORM
        "untruncated_normal" -> Distribution.UNTRUNCATED_NORMAL
        else -> Distribution.TRUNCATED_NORMAL
    }
}

private fun convertMode(mode: String): Mode {
    return when (mode) {
        "fan_in" -> Mode.FAN_IN
        "fan_out" -> Mode.FAN_OUT
        "fan_avg" -> Mode.FAN_AVG
        else -> Mode.FAN_AVG
    }
}

private fun convertToActivation(activation: String): Activations {
    return when (activation) {
        ACTIVATION_RELU -> Activations.Relu
        ACTIVATION_SIGMOID -> Activations.Sigmoid
        ACTIVATION_SOFTMAX -> Activations.Softmax
        ACTIVATION_LINEAR -> Activations.Linear
        ACTIVATION_TANH -> Activations.Tanh
        ACTIVATION_RELU6 -> Activations.Relu6
        ACTIVATION_ELU -> Activations.Elu
        ACTIVATION_SELU -> Activations.Selu
        ACTIVATION_LOG_SOFTMAX -> Activations.LogSoftmax
        ACTIVATION_EXP -> Activations.Exponential
        ACTIVATION_SOFTPLUS -> Activations.SoftPlus
        ACTIVATION_SOFTSIGN -> Activations.SoftSign
        ACTIVATION_HARD_SIGMOID -> Activations.HardSigmoid
        ACTIVATION_SWISH -> Activations.Swish
        else -> throw IllegalStateException("$activation is not supported yet!")
    }
}

/**
 * The layer creator functions should be put below.
 */

private fun createGlobalAvgPool2DLayer(name: String): Layer {
    return GlobalAvgPool2D(
        name = name
    )
}

private fun createGlobalAvgPool1DLayer(name: String): Layer {
    return GlobalAvgPool1D(
        name = name
    )
}

private fun createGlobalAvgPool3DLayer(name: String): Layer {
    return GlobalAvgPool3D(
        name = name
    )
}

private fun createGlobalMaxPool1DLayer(config: LayerConfig, name: String): Layer {
    return GlobalMaxPool1D(
        name = name
    )
}

private fun createGlobalMaxPool2DLayer(config: LayerConfig, name: String): Layer {
    return GlobalMaxPool2D(
        name = name,
    )
}

private fun createGlobalMaxPool3DLayer(config: LayerConfig, name: String): Layer {
    return GlobalMaxPool3D(
        name = name,
    )
}

private fun createAddLayer(name: String): Layer {
    return Add(
        name = name
    )
}

private fun createSubtractLayer(name: String): Layer {
    return Subtract(
        name = name
    )
}

private fun createAverageLayer(name: String): Layer {
    return Average(
        name = name
    )
}

private fun createMaximumLayer(name: String): Layer {
    return Maximum(
        name = name
    )
}

private fun createMinimumLayer(name: String): Layer {
    return Minimum(
        name = name
    )
}

private fun createMultiplyLayer(name: String): Layer {
    return Multiply(
        name = name
    )
}

private fun createConcatenateLayer(config: LayerConfig, name: String): Layer {
    return Concatenate(
        axis = config.axis!! as Int,
        name = name
    )
}

private fun createDropoutLayer(config: LayerConfig, name: String): Layer {
    return Dropout(
        keepProbability = config.rate!!.toFloat(),
        name = name
    )
}

private fun createActivationLayer(config: LayerConfig, name: String): Layer {
    return ActivationLayer(
        activation = convertToActivation(config.activation!!),
        name = name
    )
}

private fun createReLULayer(config: LayerConfig, name: String): Layer {
    return ReLU(
        maxValue = config.max_value!!.toFloat(),
        negativeSlope = config.negative_slope!!.toFloat(),
        threshold = config.threshold!!.toFloat(),
        name = name
    )
}

private fun createELULayer(config: LayerConfig, name: String): Layer {
    return ELU(
        alpha = config.alpha!!.toFloat(),
        name = name
    )
}

private fun createPReLULayer(config: LayerConfig, name: String): Layer {
    return PReLU(
        alphaInitializer = convertToInitializer(config.alpha_initializer!!),
        alphaRegularizer = convertToRegularizer(config.alpha_regularizer),
        sharedAxes = config.shared_axes!!.toIntArray(),
        name = name
    )
}

private fun createLeakyReLULayer(config: LayerConfig, name: String): Layer {
    return LeakyReLU(
        alpha = config.alpha!!.toFloat(),
        name = name
    )
}

private fun createThresholdedReLULayer(config: LayerConfig, name: String): Layer {
    return ThresholdedReLU(
        theta = config.theta!!.toFloat(),
        name = name
    )
}

private fun createSoftmaxLayer(config: LayerConfig, name: String): Layer {
    val axis = when (config.axis) {
        is Int -> listOf(config.axis)
        is List<*> -> config.axis as List<Int>
        else -> throw IllegalArgumentException("Axis must be an integer or a list of integers")
    }
    return Softmax(
        name = name,
        axis = axis
    )
}

private fun createBatchNormLayer(config: LayerConfig, name: String): Layer {
    return BatchNorm(
        axis = config.axis!! as List<Int>,
        momentum = config.momentum!!,
        center = config.center!!,
        epsilon = config.epsilon!!,
        scale = config.scale!! as Boolean,
        gammaInitializer = convertToInitializer(config.gamma_initializer!!),
        betaInitializer = convertToInitializer(config.beta_initializer!!),
        gammaRegularizer = convertToRegularizer(config.gamma_regularizer),
        betaRegularizer = convertToRegularizer(config.beta_regularizer),
        movingMeanInitializer = convertToInitializer(config.moving_mean_initializer!!),
        movingVarianceInitializer = convertToInitializer(config.moving_variance_initializer!!),
        name = name
    )
}

private fun createDenseLayer(config: LayerConfig, name: String): Layer {
    return Dense(
        outputSize = config.units!!,
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        name = name
    )
}

private fun createMaxPool1DLayer(config: LayerConfig, name: String): Layer {
    val poolSize = config.pool_size!!
    val addedOnesPoolSize = longArrayOf(1, poolSize[0].toLong(), 1)
    val strides = config.strides!!
    val addedOnesStrides = longArrayOf(1, strides[0].toLong(), 1)
    return MaxPool1D(
        poolSize = addedOnesPoolSize,
        strides = addedOnesStrides,
        padding = convertPadding(config.padding!!),
        name = name
    )
}

private fun createMaxPool2DLayer(config: LayerConfig, name: String): Layer {
    val poolSize = config.pool_size!!.toIntArray()
    val addedOnesPoolSize = IntArray(4)
    addedOnesPoolSize[0] = 1
    addedOnesPoolSize[1] = poolSize[0]
    addedOnesPoolSize[2] = poolSize[1]
    addedOnesPoolSize[3] = 1

    val strides = config.strides!!.toIntArray()
    val addedOnesStrides = IntArray(4)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = 1

    return MaxPool2D(
        poolSize = addedOnesPoolSize,
        strides = addedOnesStrides,
        padding = convertPadding(config.padding!!),
        name = name
    )
}

private fun createAvgPool1DLayer(config: LayerConfig, name: String): Layer {
    val poolSize = config.pool_size!!
    val addedOnesPoolSize = longArrayOf(1, poolSize[0].toLong(), 1)
    val strides = config.strides!!
    val addedOnesStrides = longArrayOf(1, strides[0].toLong(), 1)
    return AvgPool1D(
        poolSize = addedOnesPoolSize,
        strides = addedOnesStrides,
        padding = convertPadding(config.padding!!),
        name = name
    )
}

private fun createAvgPool2DLayer(config: LayerConfig, name: String): Layer {
    val poolSize = config.pool_size!!.toIntArray()
    val addedOnesPoolSize = IntArray(4)
    addedOnesPoolSize[0] = 1
    addedOnesPoolSize[1] = poolSize[0]
    addedOnesPoolSize[2] = poolSize[1]
    addedOnesPoolSize[3] = 1

    val strides = config.strides!!.toIntArray()
    val addedOnesStrides = IntArray(4)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = 1

    return AvgPool2D(
        poolSize = addedOnesPoolSize,
        strides = addedOnesStrides,
        padding = convertPadding(config.padding!!),
        name = name
    )
}

private fun createAvgPool3DLayer(config: LayerConfig, name: String): Layer {
    val poolSize = config.pool_size!!
    val addedOnesPoolSize = longArrayOf(1, poolSize[0].toLong(), poolSize[1].toLong(), poolSize[2].toLong(), 1)
    val strides = config.strides!!
    val addedOnesStrides = longArrayOf(1, strides[0].toLong(), strides[1].toLong(), strides[2].toLong(), 1)
    return AvgPool3D(
        poolSize = addedOnesPoolSize,
        strides = addedOnesStrides,
        padding = convertPadding(config.padding!!),
        name = name
    )
}

private fun createMaxPool3DLayer(config: LayerConfig, name: String): Layer {
    val poolSize = config.pool_size!!.toIntArray()
    val addedOnesPoolSize = IntArray(5)
    addedOnesPoolSize[0] = 1
    addedOnesPoolSize[1] = poolSize[0]
    addedOnesPoolSize[2] = poolSize[1]
    addedOnesPoolSize[3] = poolSize[2]
    addedOnesPoolSize[0] = 1

    val strides = config.strides!!.toIntArray()
    val addedOnesStrides = IntArray(5)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = strides[2]
    addedOnesStrides[4] = 1

    return MaxPool3D(
        poolSize = addedOnesPoolSize,
        strides = addedOnesStrides,
        padding = convertPadding(config.padding!!),
        name = name
    )
}

private fun convertPadding(padding: KerasPadding): ConvPadding {
    return when (padding) {
        is KerasPadding.Same -> ConvPadding.SAME
        is KerasPadding.Valid -> ConvPadding.VALID
        is KerasPadding.Full -> ConvPadding.FULL
        else -> throw UnsupportedOperationException("The $padding is not supported!")
    }
}

private fun createFlattenLayer(name: String): Layer {
    return Flatten(name = name)
}

private fun createRepeatVectorLayer(config: LayerConfig, name: String): Layer {
    return RepeatVector(name = name, n = config.n!!)
}

private fun createReshapeLayer(config: LayerConfig, name: String): Layer {
    return Reshape(name = name, targetShape = config.target_shape!!)
}

private fun createConv1DLayer(config: LayerConfig, name: String): Layer {
    val kernelSize = config.kernel_size!!.map { it.toLong() }[0]
    val strides = config.strides!!.map { it.toLong() }.toLongArray()

    val addedOnesStrides = LongArray(3)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = 1

    val dilation = config.dilation_rate!!.map { it.toLong() }.toLongArray()
    val addedOnesDilation = LongArray(3)
    addedOnesDilation[0] = 1
    addedOnesDilation[1] = dilation[0]
    addedOnesDilation[2] = 1

    return Conv1D(
        filters = config.filters!!.toLong(),
        kernelSize = kernelSize,
        strides = addedOnesStrides,
        dilations = addedOnesDilation,
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!,
        name = name
    )
}

private fun createConv2DLayer(config: LayerConfig, name: String): Layer {
    val kernelSize = config.kernel_size!!.map { it.toLong() }.toLongArray()
    val strides = config.strides!!.map { it.toLong() }.toLongArray()

    val addedOnesStrides = LongArray(4)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = 1

    val dilation = config.dilation_rate!!.map { it.toLong() }.toLongArray()
    val addedOnesDilation = LongArray(4)
    addedOnesDilation[0] = 1
    addedOnesDilation[1] = dilation[0]
    addedOnesDilation[2] = dilation[1]
    addedOnesDilation[3] = 1

    return Conv2D(
        filters = config.filters!!.toLong(),
        kernelSize = kernelSize,
        strides = addedOnesStrides,
        dilations = addedOnesDilation,
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!,
        name = name
    )
}

private fun createConv3DLayer(config: LayerConfig, name: String): Layer {
    val kernelSize = config.kernel_size!!.map { it.toLong() }.toLongArray()
    val strides = config.strides!!.map { it.toLong() }.toLongArray()

    val addedOnesStrides = LongArray(5)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = strides[2]
    addedOnesStrides[4] = 1

    val dilation = config.dilation_rate!!.map { it.toLong() }.toLongArray()
    val addedOnesDilation = LongArray(5)
    addedOnesDilation[0] = 1
    addedOnesDilation[1] = dilation[0]
    addedOnesDilation[2] = dilation[1]
    addedOnesDilation[3] = dilation[2]
    addedOnesDilation[4] = 1

    return Conv3D(
        filters = config.filters!!.toLong(),
        kernelSize = kernelSize,
        strides = addedOnesStrides,
        dilations = addedOnesDilation,
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!,
        name = name
    )
}

private fun createDepthwiseConv2DLayer(config: LayerConfig, name: String): Layer {
    val kernelSize = config.kernel_size!!.map { it.toLong() }.toLongArray()
    val strides = config.strides!!.map { it.toLong() }.toLongArray()

    val addedOnesStrides = LongArray(4)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = 1

    val dilation = config.dilation_rate!!.map { it.toLong() }.toLongArray()
    val addedOnesDilation = LongArray(4)
    addedOnesDilation[0] = 1
    addedOnesDilation[1] = dilation[0]
    addedOnesDilation[2] = dilation[1]
    addedOnesDilation[3] = 1

    return DepthwiseConv2D(
        kernelSize = kernelSize,
        strides = addedOnesStrides,
        dilations = addedOnesDilation,
        activation = convertToActivation(config.activation!!),
        depthwiseInitializer = convertToInitializer(config.depthwise_initializer!!),
        depthMultiplier = config.depth_multiplier!!,
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        depthwiseRegularizer = convertToRegularizer(config.depthwise_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!,
        name = name
    )
}

private fun createSeparableConv2DLayer(config: LayerConfig, name: String): Layer {
    val kernelSize = config.kernel_size!!.map { it.toLong() }.toLongArray()
    val strides = config.strides!!.map { it.toLong() }.toLongArray()

    val addedOnesStrides = LongArray(4)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = 1

    val dilation = config.dilation_rate!!.map { it.toLong() }.toLongArray()
    val addedOnesDilation = LongArray(4)
    addedOnesDilation[0] = 1
    addedOnesDilation[1] = dilation[0]
    addedOnesDilation[2] = dilation[1]
    addedOnesDilation[3] = 1

    return SeparableConv2D(
        filters = config.filters!!.toLong(),
        kernelSize = kernelSize,
        strides = addedOnesStrides,
        dilations = addedOnesDilation,
        activation = convertToActivation(config.activation!!),
        depthwiseInitializer = convertToInitializer(config.depthwise_initializer!!),
        pointwiseInitializer = convertToInitializer(config.pointwise_initializer!!),
        depthMultiplier = config.depth_multiplier!!,
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        depthwiseRegularizer = convertToRegularizer(config.depthwise_regularizer),
        pointwiseRegularizer = convertToRegularizer(config.pointwise_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!,
        name = name
    )
}

private fun createZeroPadding2DLayer(config: LayerConfig, name: String): Layer {
    assert(config.padding is KerasPadding.ZeroPadding2D)
    return ZeroPadding2D(
        padding = (config.padding as KerasPadding.ZeroPadding2D).padding,
        dataFormat = config.data_format,
        name = name
    )
}

private fun createCropping2DLayer(config: LayerConfig, name: String): Layer {
    val cropping = config.cropping!!.map { it.toIntArray() }.toTypedArray()
    return Cropping2D(
        cropping = cropping,
        name = name
    )
}
