/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.jetbrains.kotlinx.dl.api.core.layer.requireArraySize
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.jetbrains.kotlinx.dl.api.core.util.separableConvBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.separableConvDepthwiseKernelVarName
import org.jetbrains.kotlinx.dl.api.core.util.separableConvPointwiseKernelVarName
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.Conv2d
import org.tensorflow.op.nn.DepthwiseConv2dNative
import org.tensorflow.op.nn.DepthwiseConv2dNative.dilations

private const val DEPTHWISE_KERNEL_VARIABLE_NAME = "separable_conv2d_depthwise_kernel"
private const val POINTWISE_KERNEL_VARIABLE_NAME = "separable_conv2d_pointwise_kernel"
private const val BIAS_VARIABLE_NAME = "separable_conv2d_bias"

/**
 * 2-D convolution with separable filters.
 *
 * Performs a depthwise convolution that acts separately on channels followed by
 * a pointwise convolution that mixes channels.  Note that this is separability
 * between dimensions `[1, 2]` and `3`, not spatial separability between dimensions `1` and `2`.
 *
 * Intuitively, separable convolutions can be understood as
 * a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an Inception block.
 *
 * @property [filters] The dimensionality of the output space (i.e. the number of filters in the convolution).
 * @property [kernelSize] Two long numbers, specifying the height and width of the 2D convolution window.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
 * @property [dilations] Four numbers, specifying the dilation rate to use for dilated convolution for each dimension of input tensor.
 * @property [activation] Activation function.
 * @property [depthMultiplier] The number of depthwise convolution output channels for each input channel.
 * The total number of depthwise convolution output channels will be equal to `numberOfChannels * depthMultiplier`.
 * @property [depthwiseInitializer] An initializer for the depthwise kernel matrix.
 * @property [pointwiseInitializer] An initializer for the pointwise kernel matrix.
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [depthwiseRegularizer] Regularizer function applied to the `depthwise` kernel matrix.
 * @property [pointwiseRegularizer] Regularizer function applied to the `pointwise` kernel matrix
 * @property [biasRegularizer] Regularizer function applied to the `bias` vector.
 * @property [activityRegularizer] Regularizer function applied to the output of the layer (its "activation").
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [useBias] If true the layer uses a bias vector.
 * @property [name] Custom layer name.
 * @constructor Creates [SeparableConv2D] object.
 *
 * @since 0.2
 */
public class SeparableConv2D(
    public val filters: Long = 32,
    public val kernelSize: LongArray = longArrayOf(3, 3),
    public val strides: LongArray = longArrayOf(1, 1, 1, 1),
    public val dilations: LongArray = longArrayOf(1, 1, 1, 1),
    public val activation: Activations = Activations.Relu,
    public val depthMultiplier: Int = 1,
    public val depthwiseInitializer: Initializer = HeNormal(),
    public val pointwiseInitializer: Initializer = HeNormal(),
    public val biasInitializer: Initializer = HeUniform(),
    public val depthwiseRegularizer: Regularizer? = null,
    public val pointwiseRegularizer: Regularizer? = null,
    public val biasRegularizer: Regularizer? = null,
    public val activityRegularizer: Regularizer? = null,
    public val padding: ConvPadding = ConvPadding.SAME,
    public val useBias: Boolean = true,
    name: String = ""
) : AbstractSeparableConv(
    filtersInternal = filters,
    kernelSizeInternal = kernelSize,
    stridesInternal = strides,
    dilationsInternal = dilations,
    depthMulitplierInternal = depthMultiplier,
    activationInternal = activation,
    depthwiseInitializerInternal = depthwiseInitializer,
    pointwiseInitializerInternal = pointwiseInitializer,
    biasInitializerInternal = biasInitializer,
    depthwiseRegularizerInternal = depthwiseRegularizer,
    pointwiseRegularizerInternal = pointwiseRegularizer,
    biasRegularizerInternal = biasRegularizer,
    useBiasInternal = useBias,
    depthwiseKernelVariableName = DEPTHWISE_KERNEL_VARIABLE_NAME,
    pointwiseKernelVariableName = POINTWISE_KERNEL_VARIABLE_NAME,
    biasVariableName = BIAS_VARIABLE_NAME,
    name
), NoGradients {

    init {
        requireArraySize(kernelSize, 2, "kernelSize")
        requireArraySize(strides, 4, "strides")
        requireArraySize(dilations, 4, "dilations")
    }

    override fun separableConvImplementation(tf: Ops, input: Operand<Float>): Operand<Float> {
        val paddingName = padding.paddingName
        val depthwiseConv2DOptions: DepthwiseConv2dNative.Options = dilations(dilations.toList()).dataFormat("NHWC")

        val depthwiseOutput: Operand<Float> =
            tf.nn.depthwiseConv2dNative(
                input,
                depthwiseKernel,
                strides.toMutableList(),
                paddingName,
                depthwiseConv2DOptions
            )

        val pointwiseStrides = mutableListOf(1L, 1L, 1L, 1L)

        val conv2DOptions: Conv2d.Options = Conv2d.dataFormat("NHWC")
        return tf.nn.conv2d(depthwiseOutput, pointwiseKernel, pointwiseStrides, "VALID", conv2DOptions)
    }

    override fun defineOutputShape(inputShape: Shape): Shape {
        var rows = inputShape.size(1)
        var cols = inputShape.size(2)

        rows = convOutputLength(
            rows, kernelSize[0].toInt(), padding,
            strides[1].toInt(), dilations[1].toInt()
        )
        cols = convOutputLength(
            cols, kernelSize[1].toInt(), padding,
            strides[2].toInt(), dilations[2].toInt()
        )

        return Shape.make(inputShape.size(0), rows, cols, filters)
    }

    override fun depthwiseKernalVarName(name: String): String = separableConvDepthwiseKernelVarName(name, dim = 2)

    override fun pointwiseKernelVarName(name: String): String = separableConvPointwiseKernelVarName(name, dim = 2)

    override fun biasVarName(name: String): String = separableConvBiasVarName(name, dim = 2)

    override fun toString(): String =
        "SeparableConv2D(kernelSize=${kernelSize.contentToString()}, strides=${strides.contentToString()}, " +
                "dilations=${dilations.contentToString()}, activation=$activation, depthwiseInitializer=$depthwiseInitializer, " +
                "biasInitializer=$biasInitializer, kernelShape=$depthwiseKernelShape, padding=$padding)"
}
