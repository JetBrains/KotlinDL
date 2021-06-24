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
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.jetbrains.kotlinx.dl.api.core.util.convBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.convKernelVarName
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Squeeze
import org.tensorflow.op.nn.Conv2d

private const val KERNEL_VARIABLE_NAME = "conv1d_kernel"

private const val BIAS_VARIABLE_NAME = "conv1d_bias"

private const val EXTRA_DIM = 1L

/**
 * 1D convolution layer (e.g. convolution over audio data).
 *
 * This layer creates a convolution kernel that is convolved (actually cross-correlated)
 * with the layer input to produce a tensor of outputs.
 * Finally, the `activation` is applied to the outputs as well.
 *
 * It expects input data of size `(N, L, C)` where
 * ```
 * N - batch size
 * L - length of signal sequence
 * C - number of channels
 * ```
 *
 * @property [filters] The dimensionality of the output space (i.e. the number of filters in the convolution).
 * @property [kernelSize] Long number, specifying the width of the 1D convolution window.
 * @property [strides] Three numbers specifying the strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying stride value != 1 is incompatible with specifying `dilation` value != 1.
 * @property [dilations] Three numbers specifying the dilation rate to use for
 * dilated convolution sequence dimensions of input tensor.
 * @property [activation] Activation function.
 * @property [kernelInitializer] An initializer for the convolution kernel
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [kernelRegularizer] Regularizer function applied to the `kernel` weights matrix.
 * @property [biasRegularizer] Regularizer function applied to the `bias` vector.
 * @property [activityRegularizer] Regularizer function applied to the output of the layer (its "activation").
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [name] Custom layer name.
 * @property [useBias] If true the layer uses a bias vector.
 * @constructor Creates [Conv1D] object.
 *
 * @since 0.3
 */
public class Conv1D(
    public val filters: Long = 32,
    public val kernelSize: Long = 3,
    public val strides: LongArray = longArrayOf(1, 1, 1),
    public val dilations: LongArray = longArrayOf(1, 1, 1),
    public val activation: Activations = Activations.Relu,
    public val kernelInitializer: Initializer = HeNormal(),
    public val biasInitializer: Initializer = HeUniform(),
    public val kernelRegularizer: Regularizer? = null,
    public val biasRegularizer: Regularizer? = null,
    public val activityRegularizer: Regularizer? = null,
    public val padding: ConvPadding = ConvPadding.SAME,
    public val useBias: Boolean = true,
    name: String = "",
) : AbstractConv(
    filtersInternal = filters,
    kernelSizeInternal = longArrayOf(1, kernelSize),
    stridesInternal = longArrayOf(strides[0], 1, strides[1], strides[2]),
    dilationsInternal = longArrayOf(dilations[0], 1, dilations[1], dilations[2]),
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
        requireArraySize(strides, 3, "strides")
        requireArraySize(dilations, 3, "dilations")
    }

    /** Axis of height for which the extra dimension is added (unsqueezed) before actual
     * convolution operation and the output from actual implementation are squeezed. */
    private val squeezeAxis = Squeeze.axis(listOf(EXTRA_DIM))

    override fun kernelVarName(name: String): String = convKernelVarName(name, dim = 1)

    override fun biasVarName(name: String): String = convBiasVarName(name, dim = 1)

    override fun convImplementation(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float> {
        val options = Conv2d.dilations(dilationsInternal.toList()).dataFormat("NHWC")
        val reshapedInput = tf.expandDims(input, tf.constant(EXTRA_DIM))
        val result =
            tf.nn.conv2d(reshapedInput, kernel, stridesInternal.toMutableList(), paddingInternal.paddingName, options)
        return tf.squeeze(result, squeezeAxis)
    }

    override fun defineOutputShape(inputShape: Shape): Shape {
        val batchSize = inputShape.size(0)
        val colsCount = inputShape.size(1)

        val cols = convOutputLength(
            colsCount,
            kernelSize.toInt(),
            paddingInternal,
            strides[1].toInt(),
            dilations[1].toInt()
        )

        return Shape.make(batchSize, cols, filtersInternal)
    }

    override fun toString(): String =
        "Conv1D(" +
                "filters=$filters, " +
                "kernelSize=$kernelSize, " +
                "strides=${strides.contentToString()}, " +
                "dilation=${dilations.contentToString()}, " +
                "activation=$activation, " +
                "kernelInitializer=$kernelInitializer, " +
                "biasInitializer=$biasInitializer, " +
                "kernelShape=$kernelShape, " +
                "biasShape=$biasShape, " +
                "padding=$padding, " +
                "biasRegularizer=$biasRegularizer, " +
                "kernelRegularizer=$kernelRegularizer, " +
                "activityRegularizer=$activityRegularizer" +
                ")"
}
