/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.ZeroPadding1D
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.junit.jupiter.api.Test

internal class ZeroPadding1DTest : LayerTest() {
    private val input = arrayOf(
        arrayOf(
            floatArrayOf(0.0f, 1.0f),
            floatArrayOf(2.0f, 3.0f),
            floatArrayOf(4.0f, 5.0f)
        )
    )

    private val inputShape = input.shape.toLongArray()

    @Test
    fun default() {
        val layer = ZeroPadding1D(2, "intZeroPad1D")
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f),
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f),
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[1], inputShape[1] + 4, inputShape[2])
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }

    @Test
    fun pairPaddingTest() {
        val layer = ZeroPadding1D(Pair(2, 4), name = "pairTest")
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f),
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f),
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[1], inputShape[1] + 6, inputShape[2])
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }

    @Test
    fun arrayPaddingTest() {
        val layer = ZeroPadding1D(intArrayOf(2, 3), name = "arrayPadTest")
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f),
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f),
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[1], inputShape[1] + 5, inputShape[2])
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }
}
