/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Abstract UpSampling layer used as the base layer for all the upsampling layers.
 *
 * @property [sizeInternal] UpSampling size factors; currently, they are not used in the implementation
 * of this abstract class and each subclassed layer uses its own copy of the upsampling size factors.
 * @property [interpolationInternal] Interpolation method used for filling values (only applicable to 2D data
 * for the moment).
 */
public abstract class AbstractUpSampling(
    public val sizeInternal: IntArray,
    public val interpolationInternal: InterpolationMethod,
    name: String,
) : Layer(name) {

    override val hasActivation: Boolean
        get() = false
    override val paramCount: Int
        get() = 0
    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return upSample(tf, input)
    }

    /**
     * The actual implementation of upsampling operation which each subclassed layer needs to
     * implement. This method will then be called from [forward] method to upsample the input tensor.
     */
    protected abstract fun upSample(tf: Ops, input: Operand<Float>): Operand<Float>
}

/**
 * Repeat elements of a given tensor [value] for [repeats] times along the given [axis].
 *
 * For example, if the given tensor is equal to `[1, 2, 3]`, `repeats=2` and `axis=0`,
 * the output of this function would be `[1, 1, 2, 2, 3, 3]`.
 */
internal fun repeat(tf: Ops, value: Operand<Float>, repeats: Int, axis: Int): Operand<Float> {
    val inputShape = value.asOutput().shape()
    val splits = tf.split(tf.constant(axis), value, inputShape.size(axis))
    val multiples = tf.constant(
        IntArray(inputShape.numDimensions()) { if (it == axis) repeats else 1 }
    )
    val repeated = splits.map { tf.tile(it, multiples) }
    // The following check is due to the fact that `tf.concat` raises an error
    // if only one tensor is given as its input.
    if (repeated.size == 1)
        return repeated[0]
    return tf.concat(repeated, tf.constant(axis))
}

/**
 * Type of interpolation method.
 */
public enum class InterpolationMethod(internal val methodName: String) {
    /**
     * Nearest neighbor interpolation.
     * See: [https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation]
     */
    NEAREST("nearest"),

    /**
     * Bilinear interpolation.
     * See: [https://en.wikipedia.org/wiki/Bilinear_interpolation]
     */
    BILINEAR("bilinear"),

    /**
     * Bicubic interpolation.
     * See: [https://en.wikipedia.org/wiki/Bicubic_interpolation]
     */
    BICUBIC("bicubic"),
}
