/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Cropping layer for 1D data (e.g. audio, timeseries)
 *
 * Crops input along the second dimension (i.e. temporal dimension).
 *
 * Input shape: 3D tensor with shape `(batch_size, steps, features)`.
 *
 * Output shape: 3D tensor with shape `(batch_size, cropped_steps, features)`.
 *
 * @property [cropping] An integer array of size two (`[begin_crop, end_crop]`) indicating
 * the number of elements to remove from the beginning and end of the cropping axis.
 */
public class Cropping1D(
    public val cropping: IntArray,
    name: String = "",
) : AbstractCropping(
    croppingInternal = Array(1) { cropping },
    name = name,
) {
    init {
        require(cropping.size == 2) {
            "The cropping should be an array of size 2."
        }
    }

    override fun computeCroppedShape(inputShape: Shape): Shape {
        return Shape.make(
            inputShape.size(0),
            inputShape.size(1) - cropping[0] - cropping[1],
            inputShape.size(2)
        )
    }

    override fun crop(tf: Ops, input: Operand<Float>): Operand<Float> {
        val inputShape = input.asOutput().shape()
        val cropSize = inputShape.size(1) - cropping[0] - cropping[1]
        return tf.slice(
            input,
            tf.constant(intArrayOf(0, cropping[0], 0)),
            tf.constant(
                intArrayOf(
                    inputShape.size(0).toInt(),
                    cropSize.toInt(),
                    inputShape.size(2).toInt()
                )
            )
        )
    }

    override fun toString(): String =
        "Cropping1D(cropping=${cropping.contentToString()})"
}
