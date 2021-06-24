/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.requireArraySize
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.jetbrains.kotlinx.dl.api.core.util.convTransposeBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.convTransposeKernelVarName
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.Conv3dBackpropInput

private const val KERNEL_VARIABLE_NAME = "conv3d_transpose_kernel"

private const val BIAS_VARIABLE_NAME = "conv3d_transpose_bias"

/**
 * 3D convolution layer (e.g. spatial convolution over video frames or 3D images).
 *
 * This layer creates a convolution kernel that is convolved (actually cross-correlated)
 * with the layer input to produce a tensor of outputs.
 * Finally, the `activation` is applied to the outputs as well.
 *
 * It expects input data of size `(N, D, H, W, C)` where
 * ```
 * N - batch size
 * D - depth
 * H - height
 * W - width
 * C - number of channels
 * ```
 *
 * @property [filters] The dimensionality of the output space (i.e. the number of filters in the convolution).
 * @property [kernelSize] Three long numbers, specifying the height and width of the 3D convolution cube.
 * @property [strides] Five numbers, specifying the strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying any stride value != 1 is incompatible with specifying any `dilations` value != 1.
 * @property [dilations] Five numbers, specifying the dilation rate to use for dilated convolution for each dimension of input tensor.
 * @property [activation] Activation function.
 * @property [kernelInitializer] An initializer for the convolution kernel
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [kernelRegularizer] Regularizer function applied to the `kernel` weights matrix.
 * @property [biasRegularizer] Regularizer function applied to the `bias` vector.
 * @property [activityRegularizer] Regularizer function applied to the output of the layer (its "activation").
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [outputPadding] Three long numbers specifying the amount of padding along the height, width and depth of
 * the output tensor. Defaults to null meaning the output shape is inferred.
 * @property [name] Custom layer name.
 * @property [useBias] If true the layer uses a bias vector.
 * @constructor Creates [Conv3D] object.
 *
 * @since 0.3
 */
public class Conv3DTranspose(
    public val filters: Long = 32,
    public val kernelSize: LongArray = longArrayOf(3, 3, 3),
    public val strides: LongArray = longArrayOf(1, 1, 1, 1, 1),
    public val dilations: LongArray = longArrayOf(1, 1, 1, 1, 1),
    public val activation: Activations = Activations.Relu,
    public val kernelInitializer: Initializer = HeNormal(),
    public val biasInitializer: Initializer = HeUniform(),
    public val kernelRegularizer: Regularizer? = null,
    public val biasRegularizer: Regularizer? = null,
    public val activityRegularizer: Regularizer? = null,
    public val padding: ConvPadding = ConvPadding.SAME,
    public val outputPadding: LongArray? = null,
    public val useBias: Boolean = true,
    name: String = ""
) : AbstractConv(
    filtersInternal = filters,
    kernelSizeInternal = kernelSize,
    stridesInternal = strides,
    dilationsInternal = dilations,
    activationInternal = activation,
    kernelInitializerInternal = kernelInitializer,
    biasInitializerInternal = biasInitializer,
    kernelRegularizerInternal = kernelRegularizer,
    biasRegularizerInternal = biasRegularizer,
    activityRegularizerInternal = activityRegularizer,
    paddingInternal = padding,
    useBiasInternal = useBias,
    kernelVariableName = KERNEL_VARIABLE_NAME,
    biasVariableName = BIAS_VARIABLE_NAME,
    name = name
) {
    init {
        requireArraySize(kernelSize, 3, "kernelSize")
        requireArraySize(strides, 5, "strides")
        requireArraySize(dilations, 5, "dilations")
        if (outputPadding != null) requireArraySize(outputPadding, 3, "outputPadding")
        isTrainable = false
    }

    override fun kernelVarName(name: String): String = convTransposeKernelVarName(name, dim = 3)

    override fun biasVarName(name: String): String = convTransposeBiasVarName(name, dim = 3)

    override fun convImplementation(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float> {
        val inputSizes = convTransposeInputSizes(tf, outputShape)
        val options = Conv3dBackpropInput.dilations(dilationsInternal.toList()).dataFormat("NDHWC")
        return tf.nn.conv3dBackpropInput(inputSizes, kernel, input, stridesInternal.toMutableList(), paddingInternal.paddingName, options)
    }

    override fun defineOutputShape(inputShape: Shape): Shape {
        val batchSize = inputShape.size(0)
        val depthsCount = inputShape.size(1)
        val rowsCount = inputShape.size(2)
        val colsCount = inputShape.size(3)

        val depths = convTransposeOutputLength(
            depthsCount,
            kernelSizeInternal[0].toInt(),
            paddingInternal,
            outputPadding?.get(0)?.toInt(),
            stridesInternal[1].toInt(),
            dilationsInternal[1].toInt()
        )
        val rows = convTransposeOutputLength(
            rowsCount,
            kernelSizeInternal[1].toInt(),
            paddingInternal,
            outputPadding?.get(1)?.toInt(),
            stridesInternal[2].toInt(),
            dilationsInternal[2].toInt()
        )
        val cols = convTransposeOutputLength(
            colsCount,
            kernelSizeInternal[2].toInt(),
            paddingInternal,
            outputPadding?.get(2)?.toInt(),
            stridesInternal[3].toInt(),
            dilationsInternal[3].toInt()
        )

        return Shape.make(batchSize, depths, rows, cols, filtersInternal)
    }

    override fun toString(): String =
        "Conv3DTranspose(" +
                "filters=$filters, " +
                "kernelSize=${kernelSize.contentToString()}, " +
                "strides=${strides.contentToString()}, " +
                "dilations=${dilations.contentToString()}, " +
                "activation=$activation, " +
                "kernelInitializer=$kernelInitializer, " +
                "biasInitializer=$biasInitializer, " +
                "kernelShape=$kernelShape, " +
                "biasShape=$biasShape, " +
                "padding=$padding, " +
                "outputPadding=${outputPadding?.contentToString()} " +
                "biasRegularizer=$biasRegularizer, " +
                "kernelRegularizer=$kernelRegularizer, " +
                "activityRegularizer=$activityRegularizer" +
                ")"
}
