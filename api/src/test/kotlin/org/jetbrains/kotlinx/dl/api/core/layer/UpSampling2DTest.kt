/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.InterpolationMethod
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.UpSampling2D
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.junit.jupiter.api.Test

internal class UpSampling2DTest : LayerTest() {
    private val input = arrayOf(
        arrayOf(
            arrayOf(
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f)
            ),
            arrayOf(
                floatArrayOf(6.0f, 7.0f),
                floatArrayOf(8.0f, 9.0f),
                floatArrayOf(10.0f, 11.0f)
            )
        )
    )

    private val inputShape = input.shape.toLongArray()

    @Test
    fun default() {
        val layer = UpSampling2D()
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    floatArrayOf(0.0f, 1.0f),
                    floatArrayOf(0.0f, 1.0f),
                    floatArrayOf(2.0f, 3.0f),
                    floatArrayOf(2.0f, 3.0f),
                    floatArrayOf(4.0f, 5.0f),
                    floatArrayOf(4.0f, 5.0f)
                ),
                arrayOf(
                    floatArrayOf(0.0f, 1.0f),
                    floatArrayOf(0.0f, 1.0f),
                    floatArrayOf(2.0f, 3.0f),
                    floatArrayOf(2.0f, 3.0f),
                    floatArrayOf(4.0f, 5.0f),
                    floatArrayOf(4.0f, 5.0f)
                ),
                arrayOf(
                    floatArrayOf(6.0f, 7.0f),
                    floatArrayOf(6.0f, 7.0f),
                    floatArrayOf(8.0f, 9.0f),
                    floatArrayOf(8.0f, 9.0f),
                    floatArrayOf(10.0f, 11.0f),
                    floatArrayOf(10.0f, 11.0f)
                ),
                arrayOf(
                    floatArrayOf(6.0f, 7.0f),
                    floatArrayOf(6.0f, 7.0f),
                    floatArrayOf(8.0f, 9.0f),
                    floatArrayOf(8.0f, 9.0f),
                    floatArrayOf(10.0f, 11.0f),
                    floatArrayOf(10.0f, 11.0f)
                )
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[0], inputShape[1] * 2, inputShape[2] * 2, inputShape[3])
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }

    @Test
    fun testWithBilinearInterpolation() {
        val layer = UpSampling2D(interpolation = InterpolationMethod.BILINEAR)
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    floatArrayOf(0.0f, 1.0f),
                    floatArrayOf(0.5f, 1.5f),
                    floatArrayOf(1.5f, 2.5f),
                    floatArrayOf(2.5f, 3.5f),
                    floatArrayOf(3.5f, 4.5f),
                    floatArrayOf(4.0f, 5.0f)
                ),
                arrayOf(
                    floatArrayOf(1.5f, 2.5f),
                    floatArrayOf(2.0f, 3.0f),
                    floatArrayOf(3.0f, 4.0f),
                    floatArrayOf(4.0f, 5.0f),
                    floatArrayOf(5.0f, 6.0f),
                    floatArrayOf(5.5f, 6.5f)
                ),
                arrayOf(
                    floatArrayOf(4.5f, 5.5f),
                    floatArrayOf(5.0f, 6.0f),
                    floatArrayOf(6.0f, 7.0f),
                    floatArrayOf(7.0f, 8.0f),
                    floatArrayOf(8.0f, 9.0f),
                    floatArrayOf(8.5f, 9.5f)
                ),
                arrayOf(
                    floatArrayOf(6.0f, 7.0f),
                    floatArrayOf(6.5f, 7.5f),
                    floatArrayOf(7.5f, 8.5f),
                    floatArrayOf(8.5f, 9.5f),
                    floatArrayOf(9.5f, 10.5f),
                    floatArrayOf(10.0f, 11.0f)
                )
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[0], inputShape[1] * 2, inputShape[2] * 2, inputShape[3])
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }

    @Test
    fun testWithUnequalUpSamplingSize() {
        val layer = UpSampling2D(size = intArrayOf(2, 1))
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    floatArrayOf(0.0f, 1.0f),
                    floatArrayOf(2.0f, 3.0f),
                    floatArrayOf(4.0f, 5.0f)
                ),
                arrayOf(
                    floatArrayOf(0.0f, 1.0f),
                    floatArrayOf(2.0f, 3.0f),
                    floatArrayOf(4.0f, 5.0f)
                ),
                arrayOf(
                    floatArrayOf(6.0f, 7.0f),
                    floatArrayOf(8.0f, 9.0f),
                    floatArrayOf(10.0f, 11.0f)
                ),
                arrayOf(
                    floatArrayOf(6.0f, 7.0f),
                    floatArrayOf(8.0f, 9.0f),
                    floatArrayOf(10.0f, 11.0f)
                )
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[0], inputShape[1] * 2, inputShape[2], inputShape[3])
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }
}
