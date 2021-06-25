/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import com.beust.klaxon.Klaxon
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
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
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.*
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2L1
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.inference.keras.config.*
import java.io.File

/**
 * Saves model description as json configuration file fully compatible with the Keras TensorFlow framework.
 *
 * @param jsonConfigFile File to write model configuration.
 * @param isKerasFullyCompatible If true, it generates fully Keras-compatible configuration.
 */
public fun GraphTrainableModel.saveModelConfiguration(jsonConfigFile: File, isKerasFullyCompatible: Boolean = false) {
    val kerasModel = serializeModel(isKerasFullyCompatible)

    val jsonString2 = Klaxon()
        .converter(PaddingConverter())
        .toJsonString(kerasModel)

    jsonConfigFile.writeText(jsonString2, Charsets.UTF_8)
}

internal fun GraphTrainableModel.serializeModel(isKerasFullyCompatible: Boolean): KerasModel {
    val kerasLayers = mutableListOf<KerasLayer>()
    this.layers.forEach {
        run {
            val layer = convertToKerasLayer(it, isKerasFullyCompatible, this is Functional)
            kerasLayers.add(layer)
        }
    }

    val inputLayer = when (this) {
        is Sequential -> this.inputLayer
        is Functional -> this.inputLayer
        else -> throw UnsupportedOperationException("${this::class} is not supported yet!")
    }

    val inputShape = inputLayer.packedDims.map { it.toInt() }

    (kerasLayers.first().config as LayerConfig).batch_input_shape =
        listOf(null, inputShape[0], inputShape[1], inputShape[2]) // TODO: refactor with method for Input layer

    val config = KerasModelConfig(name = name, layers = kerasLayers)
    return KerasModel(config = config)
}

private fun convertToKerasLayer(layer: Layer, isKerasFullyCompatible: Boolean, isFunctional: Boolean): KerasLayer {
    val kerasLayer = when (layer) {
        // Core layers
        is Input -> createKerasInputLayer(layer)
        is Dense -> createKerasDenseLayer(layer, isKerasFullyCompatible)
        is ActivationLayer -> createKerasActivationLayer(layer)
        // Convolution layers
        is Conv1D -> createKerasConv1DLayer(layer, isKerasFullyCompatible)
        is Conv2D -> createKerasConv2DLayer(layer, isKerasFullyCompatible)
        is Conv3D -> createKerasConv3DLayer(layer, isKerasFullyCompatible)
        is DepthwiseConv2D -> createKerasDepthwiseConv2DLayer(layer, isKerasFullyCompatible)
        is SeparableConv2D -> createKerasSeparableConv2DLayer(layer, isKerasFullyCompatible)
        // Pooling layers
        is MaxPool1D -> createKerasMaxPool1DLayer(layer)
        is MaxPool2D -> createKerasMaxPool2DLayer(layer)
        is MaxPool3D -> createKerasMaxPool3DLayer(layer)
        is AvgPool1D -> createKerasAvgPool1DLayer(layer)
        is AvgPool2D -> createKerasAvgPool2DLayer(layer)
        is AvgPool3D -> createKerasAvgPool3DLayer(layer)
        is GlobalMaxPool1D -> createKerasGlobalMaxPool1DLayer(layer)
        is GlobalMaxPool2D -> createKerasGlobalMaxPool2DLayer(layer)
        is GlobalMaxPool3D -> createKerasGlobalMaxPool3DLayer(layer)
        is GlobalAvgPool1D -> createKerasGlobalAvgPool1DLayer(layer)
        is GlobalAvgPool2D -> createKerasGlobalAvgPool2DLayer(layer)
        is GlobalAvgPool3D -> createKerasGlobalAvgPool3DLayer(layer)
        // Recurrent layers (e.g. LSTM)
        // Normalization layers
        is BatchNorm -> createKerasBatchNormLayer(layer, isKerasFullyCompatible)
        // Regularization layers (e.g. Dropout)
        // Attention layers
        // Reshaping layers
        is Flatten -> createKerasFlattenLayer(layer)
        is RepeatVector -> createKerasRepeatVectorLayer(layer)
        is ZeroPadding2D -> createKerasZeroPadding2DLayer(layer)
        is UpSampling1D -> createKerasUpSampling1DLayer(layer)
        is UpSampling2D -> createKerasUpSampling2DLayer(layer)
        is UpSampling3D -> createKerasUpSampling3DLayer(layer)
        // Merging layers
        is Add -> createKerasAddLayer(layer)
        is Maximum -> createKerasMaximumLayer(layer)
        is Minimum -> createKerasMinimumLayer(layer)
        is Subtract -> createKerasSubtractLayer(layer)
        is Multiply -> createKerasMultiplyLayer(layer)
        is Average -> createKerasAverageLayer(layer)
        is Concatenate -> createKerasConcatenateLayer(layer)
        // Locally-connected layers
        // Activation layers
        is Softmax -> createKerasSoftmaxLayer(layer)
        is PReLU -> createKerasPReLULayer(layer, isKerasFullyCompatible)
        is ReLU -> createKerasReLULayer(layer)
        is ELU -> createKerasELULayer(layer)
        is LeakyReLU -> createKerasLeakyReLULayer(layer)
        is ThresholdedReLU -> createKerasThresholdedReLULayer(layer)
        else -> throw IllegalStateException("${layer.name} with type ${layer::class.simpleName} is not supported yet!")
    }

    if (isFunctional) {
        if (kerasLayer.class_name.equals("InputLayer")) {
            kerasLayer.inbound_nodes = listOf()
        } else {
            kerasLayer.inbound_nodes = listOf()

            for (inboundLayer in layer.inboundLayers) {
                if (kerasLayer.inbound_nodes!!.isEmpty()) {
                    kerasLayer.inbound_nodes = listOf(mutableListOf<Any>())
                }

                (kerasLayer.inbound_nodes as List<MutableList<MutableList<Any>>>)[0].add(
                    mutableListOf(
                        inboundLayer.name,
                        0,
                        0,
                        mapOf<Any, Any>()
                    )
                )
            }
        }
    }

    return kerasLayer
}

private fun convertToKerasRegularizer(regularizer: Regularizer?): KerasRegularizer? {
    return if (regularizer != null) {
        val className = "L1L2"
        regularizer as L2L1
        val config = KerasRegularizerConfig(l1 = regularizer.l1.toDouble(), l2 = regularizer.l2.toDouble())
        KerasRegularizer(class_name = className, config = config)
    } else {
        null
    }
}

private fun convertToKerasInitializer(initializer: Initializer, isKerasFullyCompatible: Boolean): KerasInitializer? {
    val className: String
    val config: KerasInitializerConfig
    if (isKerasFullyCompatible) {
        val (_className, _config) = when (initializer) {
            is GlorotUniform -> convertToVarianceScalingInitializer(initializer as VarianceScaling)
            is GlorotNormal -> convertToVarianceScalingInitializer(initializer as VarianceScaling)
            is HeNormal -> convertToVarianceScalingInitializer(initializer as VarianceScaling)
            is HeUniform -> convertToVarianceScalingInitializer(initializer as VarianceScaling)
            is LeCunNormal -> convertToVarianceScalingInitializer(initializer as VarianceScaling)
            is LeCunUniform -> convertToVarianceScalingInitializer(initializer as VarianceScaling)
            is RandomUniform -> convertToRandomUniformInitializer(initializer)
            is Identity -> convertToIdentityInitializer(initializer)
            else -> throw IllegalStateException("${initializer::class.simpleName} is not supported yet!")
        }

        className = _className
        config = _config
    } else {
        className = when (initializer) {
            is GlorotUniform -> INITIALIZER_GLOROT_UNIFORM
            is GlorotNormal -> INITIALIZER_GLOROT_NORMAL
            is HeNormal -> INITIALIZER_HE_NORMAL
            is HeUniform -> INITIALIZER_HE_UNIFORM
            is LeCunNormal -> INITIALIZER_LECUN_NORMAL
            is LeCunUniform -> INITIALIZER_LECUN_UNIFORM
            is Identity -> INITIALIZER_IDENTITY
            else -> throw IllegalStateException("${initializer::class.simpleName} is not supported yet!")
        }
        config = KerasInitializerConfig(seed = 12)
    }

    return KerasInitializer(class_name = className, config = config)
}

private fun convertToRandomUniformInitializer(initializer: RandomUniform): Pair<String, KerasInitializerConfig> {
    return Pair(
        INITIALIZER_RANDOM_UNIFORM, KerasInitializerConfig(
            minval = initializer.minVal.toDouble(),
            maxval = initializer.maxVal.toDouble(),
            seed = initializer.seed.toInt()
        )
    )
}

private fun convertToVarianceScalingInitializer(initializer: VarianceScaling): Pair<String, KerasInitializerConfig> {
    return Pair(
        INITIALIZER_VARIANCE_SCALING, KerasInitializerConfig(
            seed = initializer.seed.toInt(),
            scale = initializer.scale,
            mode = convertMode(initializer.mode),
            distribution = convertDistribution(initializer.distribution)
        )
    )
}

private fun convertToIdentityInitializer(initializer: Identity): Pair<String, KerasInitializerConfig> {
    return Pair(
        INITIALIZER_IDENTITY,
        KerasInitializerConfig(
            gain = initializer.gain.toDouble()
        )
    )
}

private fun convertDistribution(distribution: Distribution): String {
    return when (distribution) {
        Distribution.TRUNCATED_NORMAL -> "truncated_normal"
        Distribution.UNIFORM -> "uniform"
        Distribution.UNTRUNCATED_NORMAL -> "untruncated_normal"
    }
}

private fun convertMode(mode: Mode): String {
    return when (mode) {
        Mode.FAN_IN -> "fan_in"
        Mode.FAN_OUT -> "fan_out"
        Mode.FAN_AVG -> "fan_avg"
    }
}

private fun convertToKerasPadding(padding: ConvPadding): KerasPadding {
    return when (padding) {
        ConvPadding.SAME -> KerasPadding.Same
        ConvPadding.VALID -> KerasPadding.Valid
        ConvPadding.FULL -> KerasPadding.Full
    }
}

private fun convertToKerasActivation(activation: Activations): String? {
    return when (activation) {
        Activations.Relu -> ACTIVATION_RELU
        Activations.Sigmoid -> ACTIVATION_SIGMOID
        Activations.Softmax -> ACTIVATION_SOFTMAX
        Activations.Linear -> ACTIVATION_LINEAR
        Activations.Tanh -> ACTIVATION_TANH
        Activations.Relu6 -> ACTIVATION_RELU6
        Activations.Elu -> ACTIVATION_ELU
        Activations.Selu -> ACTIVATION_SELU
        Activations.LogSoftmax -> ACTIVATION_LOG_SOFTMAX
        Activations.Exponential -> ACTIVATION_EXP
        Activations.SoftPlus -> ACTIVATION_SOFTPLUS
        Activations.SoftSign -> ACTIVATION_SOFTSIGN
        Activations.HardSigmoid -> ACTIVATION_HARD_SIGMOID
        Activations.Swish -> ACTIVATION_SWISH
    }
}

/**
 * The layer creator functions for Keras should be put below.
 */

private fun createKerasGlobalAvgPool2DLayer(layer: GlobalAvgPool2D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_GLOBAL_AVG_POOL_2D, config = configX)
}

private fun createKerasGlobalAvgPool1DLayer(layer: GlobalAvgPool1D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_GLOBAL_AVG_POOL_1D, config = configX)
}

private fun createKerasGlobalMaxPool1DLayer(layer: GlobalMaxPool1D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_GLOBAL_MAX_POOL_1D, config = configX)
}

private fun createKerasGlobalMaxPool2DLayer(layer: GlobalMaxPool2D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable,
    )
    return KerasLayer(class_name = LAYER_GLOBAL_MAX_POOL_2D, config = configX)
}

private fun createKerasGlobalMaxPool3DLayer(layer: GlobalMaxPool3D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable,
    )
    return KerasLayer(class_name = LAYER_GLOBAL_MAX_POOL_3D, config = configX)
}

private fun createKerasGlobalAvgPool3DLayer(layer: GlobalAvgPool3D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_GLOBAL_AVG_POOL_3D, config = configX)
}

private fun createKerasAddLayer(layer: Add): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_ADD, config = configX)
}

private fun createKerasSubtractLayer(layer: Subtract): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_SUBTRACT, config = configX)
}

private fun createKerasMinimumLayer(layer: Minimum): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_MINIMUM, config = configX)
}

private fun createKerasMaximumLayer(layer: Maximum): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_MAXIMUM, config = configX)
}

private fun createKerasAverageLayer(layer: Average): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_AVERAGE, config = configX)
}

private fun createKerasMultiplyLayer(layer: Multiply): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_MULTIPLY, config = configX)
}

private fun createKerasActivationLayer(layer: ActivationLayer): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        activation = convertToKerasActivation(layer.activation),
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_ACTIVATION, config = configX)
}

private fun createKerasPReLULayer(layer: PReLU, isKerasFullyCompatible: Boolean): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        alpha_initializer = convertToKerasInitializer(layer.alphaInitializer, isKerasFullyCompatible),
        alpha_regularizer = convertToKerasRegularizer(layer.alphaRegularizer),
        shared_axes = layer.sharedAxes?.toList(),
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_PRELU, config = configX)
}

private fun createKerasSoftmaxLayer(layer: Softmax): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        axis = layer.axis,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_SOFTMAX, config = configX)
}

private fun createKerasReLULayer(layer: ReLU): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        max_value = layer.maxValue?.toDouble(),
        negative_slope = layer.negativeSlope.toDouble(),
        threshold = layer.threshold.toDouble(),
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_RELU, config = configX)
}

private fun createKerasELULayer(layer: ELU): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        alpha = layer.alpha.toDouble(),
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_ELU, config = configX)
}

private fun createKerasLeakyReLULayer(layer: LeakyReLU): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        alpha = layer.alpha.toDouble(),
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_LEAKY_RELU, config = configX)
}

private fun createKerasThresholdedReLULayer(layer: ThresholdedReLU): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        theta = layer.theta.toDouble(),
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_THRESHOLDED_RELU, config = configX)
}

private fun createKerasBatchNormLayer(layer: BatchNorm, isKerasFullyCompatible: Boolean): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable,
        axis = layer.axis,
        momentum = layer.momentum,
        center = layer.center,
        epsilon = layer.epsilon,
        scale = layer.scale,
        gamma_initializer = convertToKerasInitializer(layer.gammaInitializer, isKerasFullyCompatible),
        beta_initializer = convertToKerasInitializer(layer.betaInitializer, isKerasFullyCompatible),
        moving_mean_initializer = convertToKerasInitializer(layer.movingMeanInitializer, isKerasFullyCompatible),
        moving_variance_initializer = convertToKerasInitializer(
            layer.movingVarianceInitializer,
            isKerasFullyCompatible
        ),
        beta_regularizer = convertToKerasRegularizer(layer.betaRegularizer),
        gamma_regularizer = convertToKerasRegularizer(layer.gammaRegularizer),
        //activity_regularizer = convertToKerasRegularizer(layer.activityRegularizer),
    )
    return KerasLayer(class_name = LAYER_BATCH_NORM, config = configX)
}

private fun createKerasInputLayer(layer: Input): KerasLayer {
    val shape = mutableListOf<Int?>()
    shape.add(null)
    layer.packedDims.map { it.toInt() }.forEach { shape.add(it) }

    val config = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        sparse = false,
        batch_input_shape = shape,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_INPUT, config = config)
}

private fun createKerasDenseLayer(layer: Dense, isKerasFullyCompatible: Boolean): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        units = layer.outputSize,
        name = layer.name,
        use_bias = layer.useBias,
        trainable = layer.isTrainable,
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasInitializer(layer.kernelInitializer, isKerasFullyCompatible),
        bias_initializer = convertToKerasInitializer(layer.biasInitializer, isKerasFullyCompatible),
        kernel_regularizer = convertToKerasRegularizer(layer.kernelRegularizer),
        bias_regularizer = convertToKerasRegularizer(layer.biasRegularizer),
        activity_regularizer = convertToKerasRegularizer(layer.activityRegularizer),
    )
    return KerasLayer(class_name = LAYER_DENSE, config = configX)
}

private fun createKerasMaxPool1DLayer(layer: MaxPool1D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        pool_size = listOf(layer.poolSize[1].toInt()),
        strides = listOf(layer.strides[1].toInt()),
        padding = convertToKerasPadding(layer.padding),
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_MAX_POOL_1D, config = configX)
}

private fun createKerasMaxPool2DLayer(layer: MaxPool2D): KerasLayer {
    val poolSize = mutableListOf(layer.poolSize[1], layer.poolSize[2])
    val strides = mutableListOf(layer.strides[1], layer.strides[2])
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable,
        padding = convertToKerasPadding(layer.padding),
        pool_size = poolSize,
        strides = strides
    )
    return KerasLayer(class_name = LAYER_MAX_POOL_2D, config = configX)
}

private fun createKerasAvgPool1DLayer(layer: AvgPool1D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        pool_size = listOf(layer.poolSize[1].toInt()),
        strides = listOf(layer.strides[1].toInt()),
        padding = convertToKerasPadding(layer.padding),
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_AVG_POOL_1D, config = configX)
}

private fun createKerasMaxPool3DLayer(layer: MaxPool3D): KerasLayer {
    val poolSize = mutableListOf(layer.poolSize[1], layer.poolSize[3])
    val strides = mutableListOf(layer.strides[1], layer.strides[3])
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable,
        padding = convertToKerasPadding(layer.padding),
        pool_size = poolSize,
        strides = strides
    )
    return KerasLayer(class_name = LAYER_MAX_POOL_3D, config = configX)
}

private fun createKerasAvgPool2DLayer(layer: AvgPool2D): KerasLayer {
    val poolSize = mutableListOf(layer.poolSize[1], layer.poolSize[2])
    val strides = mutableListOf(layer.strides[1], layer.strides[2])
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable,
        padding = convertToKerasPadding(layer.padding),
        pool_size = poolSize,
        strides = strides
    )
    return KerasLayer(class_name = LAYER_AVG_POOL_2D, config = configX)
}

private fun createKerasAvgPool3DLayer(layer: AvgPool3D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        pool_size = layer.poolSize.slice(1..3).map { it.toInt() },
        strides = layer.strides.slice(1..3).map { it.toInt() },
        padding = convertToKerasPadding(layer.padding),
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_AVG_POOL_3D, config = configX)
}

private fun createKerasFlattenLayer(layer: Flatten): KerasLayer {
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_FLATTEN, config = configX)
}

private fun createKerasRepeatVectorLayer(layer: RepeatVector): KerasLayer {
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        trainable = layer.isTrainable,
        name = layer.name,
        n = layer.n
    )
    return KerasLayer(class_name = LAYER_REPEAT_VECTOR, config = configX)
}

private fun createKerasConcatenateLayer(layer: Concatenate): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        axis = layer.axis,
        name = layer.name,
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_CONCATENATE, config = configX)
}

private fun createKerasConv1DLayer(layer: Conv1D, isKerasFullyCompatible: Boolean): KerasLayer {
    val configX = LayerConfig(
        filters = layer.filters.toInt(),
        kernel_size = listOf(layer.kernelSize.toInt()),
        strides = listOf(layer.strides[1].toInt()),
        dilation_rate = listOf(layer.dilations[1].toInt()),
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasInitializer(layer.kernelInitializer, isKerasFullyCompatible),
        bias_initializer = convertToKerasInitializer(layer.biasInitializer, isKerasFullyCompatible),
        kernel_regularizer = convertToKerasRegularizer(layer.kernelRegularizer),
        bias_regularizer = convertToKerasRegularizer(layer.biasRegularizer),
        activity_regularizer = convertToKerasRegularizer(layer.activityRegularizer),
        padding = convertToKerasPadding(layer.padding),
        trainable = layer.isTrainable,
        use_bias = layer.useBias,
        name = layer.name
    )
    return KerasLayer(class_name = LAYER_CONV1D, config = configX)
}

private fun createKerasConv2DLayer(layer: Conv2D, isKerasFullyCompatible: Boolean): KerasLayer {
    val kernelSize = layer.kernelSize.map { it.toInt() }.toList()
    val configX = LayerConfig(
        filters = layer.filters.toInt(),
        kernel_size = kernelSize,
        strides = listOf(layer.strides[1].toInt(), layer.strides[2].toInt()),
        dilation_rate = listOf(layer.dilations[1].toInt(), layer.dilations[2].toInt()),
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasInitializer(layer.kernelInitializer, isKerasFullyCompatible),
        bias_initializer = convertToKerasInitializer(layer.biasInitializer, isKerasFullyCompatible),
        kernel_regularizer = convertToKerasRegularizer(layer.kernelRegularizer),
        bias_regularizer = convertToKerasRegularizer(layer.biasRegularizer),
        activity_regularizer = convertToKerasRegularizer(layer.activityRegularizer),
        padding = convertToKerasPadding(layer.padding),
        trainable = layer.isTrainable,
        use_bias = layer.useBias,
        name = layer.name
    )
    return KerasLayer(class_name = LAYER_CONV2D, config = configX)
}

private fun createKerasConv3DLayer(layer: Conv3D, isKerasFullyCompatible: Boolean): KerasLayer {
    val kernelSize = layer.kernelSize.map { it.toInt() }.toList()
    val configX = LayerConfig(
        filters = layer.filters.toInt(),
        kernel_size = kernelSize,
        strides = listOf(layer.strides[1].toInt(), layer.strides[2].toInt(), layer.strides[3].toInt()),
        dilation_rate = listOf(layer.dilations[1].toInt(), layer.dilations[2].toInt(), layer.dilations[3].toInt()),
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasInitializer(layer.kernelInitializer, isKerasFullyCompatible),
        bias_initializer = convertToKerasInitializer(layer.biasInitializer, isKerasFullyCompatible),
        kernel_regularizer = convertToKerasRegularizer(layer.kernelRegularizer),
        bias_regularizer = convertToKerasRegularizer(layer.biasRegularizer),
        activity_regularizer = convertToKerasRegularizer(layer.activityRegularizer),
        padding = convertToKerasPadding(layer.padding),
        name = layer.name,
        use_bias = layer.useBias
    )
    return KerasLayer(class_name = LAYER_CONV3D, config = configX)
}

private fun createKerasDepthwiseConv2DLayer(layer: DepthwiseConv2D, isKerasFullyCompatible: Boolean): KerasLayer {
    val configX = LayerConfig(
        kernel_size = layer.kernelSize.map { it.toInt() },
        strides = listOf(layer.strides[1].toInt(), layer.strides[2].toInt()),
        dilation_rate = listOf(layer.dilations[1].toInt(), layer.dilations[2].toInt()),
        activation = convertToKerasActivation(layer.activation),
        depthwise_initializer = convertToKerasInitializer(layer.depthwiseInitializer, isKerasFullyCompatible),
        depth_multiplier = layer.depthMultiplier,
        bias_initializer = convertToKerasInitializer(layer.biasInitializer, isKerasFullyCompatible),
        padding = convertToKerasPadding(layer.padding),
        trainable = layer.isTrainable,
        use_bias = layer.useBias,
        name = layer.name
    )
    return KerasLayer(class_name = LAYER_DEPTHWISE_CONV2D, configX)
}

private fun createKerasSeparableConv2DLayer(layer: SeparableConv2D, isKerasFullyCompatible: Boolean): KerasLayer {
    val configX = LayerConfig(
        filters = layer.filters.toInt(),
        kernel_size = layer.kernelSize.map { it.toInt() },
        strides = listOf(layer.strides[1].toInt(), layer.strides[2].toInt()),
        dilation_rate = listOf(layer.dilations[1].toInt(), layer.dilations[2].toInt()),
        activation = convertToKerasActivation(layer.activation),
        depthwise_initializer = convertToKerasInitializer(layer.depthwiseInitializer, isKerasFullyCompatible),
        pointwise_initializer = convertToKerasInitializer(layer.pointwiseInitializer, isKerasFullyCompatible),
        depth_multiplier = layer.depthMultiplier,
        bias_initializer = convertToKerasInitializer(layer.biasInitializer, isKerasFullyCompatible),
        padding = convertToKerasPadding(layer.padding),
        trainable = layer.isTrainable,
        use_bias = layer.useBias,
        name = layer.name
    )
    return KerasLayer(class_name = LAYER_SEPARABLE_CONV2D, config = configX)
}

private fun createKerasZeroPadding2DLayer(layer: ZeroPadding2D): KerasLayer {
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        padding = KerasPadding.ZeroPadding2D(layer.padding),
        trainable = layer.isTrainable
    )
    return KerasLayer(class_name = LAYER_ZERO_PADDING_2D, config = configX)
}

private fun createKerasUpSampling1DLayer(layer: UpSampling1D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        size = layer.size,
        name = layer.name,
        trainable = layer.isTrainable,
    )
    return KerasLayer(class_name = LAYER_UP_SAMPLING_1D,  config = configX)
}

private fun createKerasUpSampling2DLayer(layer: UpSampling2D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        size = layer.size.toList(),
        interpolation = layer.interpolation.methodName,
        name = layer.name,
        trainable = layer.isTrainable,
    )
    return KerasLayer(class_name = LAYER_UP_SAMPLING_2D,  config = configX)
}

private fun createKerasUpSampling3DLayer(layer: UpSampling3D): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        size = layer.size.toList(),
        name = layer.name,
        trainable = layer.isTrainable,
    )
    return KerasLayer(class_name = LAYER_UP_SAMPLING_3D,  config = configX)
}
