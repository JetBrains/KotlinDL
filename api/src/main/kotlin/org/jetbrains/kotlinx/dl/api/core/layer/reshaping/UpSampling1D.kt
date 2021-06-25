/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Squeeze

/**
 * Upsampling layer for 1D input.
 *
 * Repeats each step of the second axis [size] times.
 *
 * Input shape: 3D tensor with shape `(batch_size, steps, features)`.
 *
 * Output shape: 3D tensor with shape `(batch_size, steps * size, features)`.
 *
 * @property [size] Upsampling factor (i.e. number of repeats).
 */
public class UpSampling1D(
    public val size: Int = 2,
    name: String = "",
) : AbstractUpSampling(
    sizeInternal = intArrayOf(size),
    interpolationInternal = InterpolationMethod.NEAREST,
    name = name,
) {
    init {
        require(size > 0) {
            "The upsampling size should be a positive integer."
        }
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return Shape.make(
            inputShape.size(0),
            inputShape.size(1) * size,
            inputShape.size(2)
        )
    }

    protected override fun upSample(tf: Ops, input: Operand<Float>): Operand<Float> {
        return repeat(tf, input, repeats = size, axis = 1)
    }

    override fun toString(): String =
        "UpSampling1D(size=$size)"
}
