/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.UpSampling1D
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.junit.jupiter.api.Test

internal class UpSampling1DTest : LayerTest() {
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
        val layer = UpSampling1D()
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f),
                floatArrayOf(4.0f, 5.0f)
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[0], inputShape[1] * 2, inputShape[2])
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }

    @Test
    fun testWithNonDefaultUpSamplingSize() {
        val layer = UpSampling1D(size = 3)
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f),
                floatArrayOf(4.0f, 5.0f),
                floatArrayOf(4.0f, 5.0f)
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[0], inputShape[1] * 3, inputShape[2])
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }
}
