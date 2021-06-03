/*
* Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
* Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
*/

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Squeeze

/**
 * Average pooling operation for 1D temporal data (e.g. audio, timseries).
 *
 * Downsamples the input by taking the average over a temporal window of size [poolSize].
 *
 * @property [poolSize] Size of the temporal pooling window.
 * @property [strides] The amount of shift for pooling window in each pooling step. If
 * `null`, it will default to [poolSize].
 * @property [padding] Padding strategy; can be either of [ConvPadding.VALID] which means no
 * padding, or [ConvPadding.SAME] which means padding the input equally such that the output
 * has the same dimension as the input.
 * @property [dataFormat] Data format of input; can be either of [CHANNELS_LAST] or [CHANNELS_FIRST].
 */
public class AvgPool1D(
    public val poolSize: Int = 2,
    public val strides: Int? = null,
    public val padding: ConvPadding = ConvPadding.VALID,
    public val dataFormat: String = CHANNELS_LAST,
    name: String = ""
) : Layer(name) {

    override val hasActivation: Boolean
        get() = false
    override val paramCount: Int
        get() = 0
    override val weights: Map<String, Array<*>>
        get() = emptyMap()

    init {
        require(dataFormat == CHANNELS_LAST || dataFormat == CHANNELS_FIRST) {
            "The dataFormat should be either of \"$CHANNELS_LAST\" or \"$CHANNELS_FIRST\"."
        }

        require(padding == ConvPadding.VALID || padding == ConvPadding.SAME) {
            "The padding should be either of ${ConvPadding.VALID} or ${ConvPadding.SAME}."
        }
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        var steps = if(dataFormat == CHANNELS_LAST) inputShape.size(1) else inputShape.size(2)
        val strideValue = strides ?: poolSize
        steps = convOutputLength(steps, poolSize, padding, strideValue)
        return if (dataFormat == CHANNELS_LAST) {
            Shape.make(inputShape.size(0), steps, inputShape.size(2))
        } else {
            Shape.make(inputShape.size(0), inputShape.size(1), steps)
        }
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val expandAxis = if (dataFormat == CHANNELS_LAST) 2 else 3
        val tfInput = tf.expandDims(input, tf.constant(expandAxis))
        val tfPoolSize = longArrayOf(1, 1, 1, 1)
        val tfStrides = longArrayOf(1, 1, 1, 1)
        tfPoolSize[expandAxis-1] = poolSize.toLong()
        tfStrides[expandAxis-1] = (strides ?: poolSize).toLong()
        val tfPadding = padding.paddingName

        val avgPool = tf.nn.avgPool(
            tfInput,
            tfPoolSize.toList(),
            tfStrides.toList(),
            tfPadding
        )
        return tf.squeeze(avgPool, Squeeze.axis(listOf(expandAxis.toLong())))
    }

    override fun toString(): String =
        "AvgPool1D(poolSize=$poolSize, strides=$strides, padding=$padding, dataFormat=$dataFormat)"
}
