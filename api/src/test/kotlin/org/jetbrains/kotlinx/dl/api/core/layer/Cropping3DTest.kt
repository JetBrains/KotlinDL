/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Cropping3D
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.junit.jupiter.api.Test

internal class Cropping3DTest : LayerTest() {
    private val input = arrayOf(
        arrayOf(
            arrayOf(
                arrayOf(
                    floatArrayOf(1.0f, 2.0f, 3.0f),
                    floatArrayOf(4.0f, 5.0f, 6.0f),
                    floatArrayOf(7.0f, 8.0f, 9.0f),
                    floatArrayOf(10.0f, 11.0f, 12.0f),
                ),
                arrayOf(
                    floatArrayOf(11.0f, 12.0f, 13.0f),
                    floatArrayOf(14.0f, 15.0f, 16.0f),
                    floatArrayOf(17.0f, 18.0f, 19.0f),
                    floatArrayOf(20.0f, 21.0f, 22.0f),
                ),
            ),
            arrayOf(
                arrayOf(
                    floatArrayOf(-1.0f, -2.0f, -3.0f),
                    floatArrayOf(-4.0f, -5.0f, -6.0f),
                    floatArrayOf(-7.0f, -8.0f, -9.0f),
                    floatArrayOf(-10.0f, -11.0f, -12.0f),
                ),
                arrayOf(
                    floatArrayOf(-11.0f, -12.0f, -13.0f),
                    floatArrayOf(-14.0f, -15.0f, -16.0f),
                    floatArrayOf(-17.0f, -18.0f, -19.0f),
                    floatArrayOf(-20.0f, -21.0f, -22.0f),
                )
            )
        )
    )

    private val inputShape = input.shape.toLongArray()

    @Test
    fun testWithCroppingSize() {
        val layer = Cropping3D(
            cropping = arrayOf(intArrayOf(1, 0), intArrayOf(0, 1), intArrayOf(1, 1))
        )
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    arrayOf(
                        floatArrayOf(-4.0f, -5.0f, -6.0f),
                        floatArrayOf(-7.0f, -8.0f, -9.0f),
                    )
                )
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(
            inputShape[0], inputShape[1] - 1, inputShape[2] - 1, inputShape[3] - 1 - 1, inputShape[4]
        )
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }
}
