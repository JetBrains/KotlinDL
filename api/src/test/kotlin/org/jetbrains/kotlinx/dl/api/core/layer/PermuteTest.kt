/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Permute
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.junit.jupiter.api.Test

internal class PermuteTest : LayerTest() {
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
        val layer = Permute(dims = intArrayOf(2, 1), name = "permuteTest")
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(0.0f, 2.0f, 4.0f),
                floatArrayOf(1.0f, 3.0f, 5.0f)
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[0], inputShape[2], inputShape[1])
        assertLayerComputedOutputShape(layer, inputShape, expectedShape)
    }
}
