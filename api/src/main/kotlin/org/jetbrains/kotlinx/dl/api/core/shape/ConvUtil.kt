/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.shape

import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import kotlin.math.max

/**
 * Calculates output length after applying convolution operation on single axis.
 */
internal fun convOutputLength(
    inputLength: Long,
    filterSize: Int,
    padding: ConvPadding,
    stride: Int,
    dilation: Int = 1
): Long {
    val dilatedFilterSize = filterSize + (filterSize - 1) * (dilation - 1)
    val outputLength = when (padding) {
        ConvPadding.SAME -> inputLength
        ConvPadding.VALID -> inputLength - dilatedFilterSize + 1
        ConvPadding.FULL -> inputLength + dilatedFilterSize - 1
    }
    return ((outputLength + stride - 1).toFloat() / stride).toLong()
}

/**
 * Calculates output length after applying transposed convolution operation on single axis.
 */
internal fun convTransposeOutputLength(
    inputLength: Long,
    filterSize: Int,
    padding: ConvPadding,
    outputPadding: Int?,
    stride: Int,
    dilation: Int
): Long {
    val dilatedKernelSize = filterSize + (filterSize - 1) * (dilation - 1)
    return  if (outputPadding == null) {
        when (padding) {
            ConvPadding.VALID -> inputLength * stride + max(dilatedKernelSize - stride, 0)
            ConvPadding.FULL -> inputLength * stride - (stride + dilatedKernelSize - 2)
            ConvPadding.SAME -> inputLength * stride
        }
    }
    else {
        val pad = when (padding) {
            ConvPadding.SAME -> dilatedKernelSize / 2
            ConvPadding.VALID -> 0
            ConvPadding.FULL -> dilatedKernelSize - 1
        }
        (inputLength - 1) * stride + dilatedKernelSize - 2 * pad + outputPadding
    }
}

internal fun convTransposeInputSizes(tf: Ops, outputShape: TensorShape): Operand<Int> =
    tf.stack(outputShape.dims().map(Long::toInt).map(tf::constant))