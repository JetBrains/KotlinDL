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
 * Zero-padding layer for 3D input (e.g. video).
 *
 * This layer can add zeros in the rows, cols and depth of an video tensor.
 *
 * @property [padding] 6 numbers  interpreted as `(left_dim1_pad, right_dim1_pad, left_dim2_pad, right_dim2_pad, left_dim3_pad, right_dim3_pad)`.
 *
 * @see [Tensorflow implementation](https://github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/layers/convolutional.py#L2890)
 */
public class ZeroPadding3D : Layer {
    //TODO add dataFormat support
    public val padding: IntArray
    private lateinit var inputShape: Shape

    /**
     * Constructs an instance of ZeroPadding3D layer
     * @param [padding] symmetric padding applied to width, height and depth (same on all sides)
     * @param [name] layer name
     */
    public constructor(
        padding: Int,
        name: String = ""
    ) : this(
        IntArray(6) { padding },
        name
    )

    /**
     * Constructs an instance of ZeroPadding3D layer
     * @param [padding] triple of padding values - [padding.first] represents vertical padding (applied to top and
     * bottom of image, and [padding.second] is horizontal padding (left and right sides), [padding.third] is depth padding
     * @param [name] layer name
     */
    public constructor(
        padding: Triple<Int, Int, Int>,
        name: String = ""
    ) : this(
        intArrayOf(padding.first, padding.first, padding.second, padding.second, padding.third, padding.third),
        name
    )

    /**
     * Constructs an instance of ZeroPadding3D layer
     * @param [padding] list of pair of padding values [padding[0]] represents the first pair(applied to vertical),
     * [padding[1]] is horizontal padding, [padding[2]] is the depth padding.
     * @param [name] layer name
     */
    public constructor(
        padding: Array<Pair<Int, Int>>,
        name: String = ""
    ) : this(
        intArrayOf(
            padding[0].first, padding[0].second,
            padding[1].first, padding[1].second,
            padding[2].first, padding[2].second
        ),
        name
    )

    /**
     * Constructs an instance of ZeroPadding3D layer
     * @param [padding] list of padding values. Size of list must be equal to 6. Those list values maps to
     * the following paddings:
     * padding[0] -> top padding,
     * padding[1] -> bottom padding,
     * padding[2] -> left padding,
     * padding[3] -> right padding
     * padding[4] -> front padding
     * padding[5] -> back padding
     * @param [name] layer name
     */
    public constructor(padding: IntArray, name: String = "") : super(name) {
        require(padding.size == 6)
        this.padding = padding
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        this.inputShape = inputShape
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        val dim1 = inputShape.size(1) + padding[0] + padding[1];
        val dim2 = inputShape.size(2) + padding[2] + padding[3];
        val dim3 = inputShape.size(3) + padding[4] + padding[5];
        return Shape.make(inputShape.size(0), dim1, dim2, dim3, inputShape.size(4))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val paddingOperand = tf.constant(paddingArrayToTfFormat())
        val constantValue = tf.constant(0f)
        return tf.pad(input, paddingOperand, constantValue)
    }

    private fun paddingArrayToTfFormat(): Array<IntArray> {
        val paddingFirstDim: IntArray
        val paddingSecondDim: IntArray
        val paddingThirdDim: IntArray

        when (padding.size) {
            6 -> {
                paddingFirstDim = intArrayOf(padding[0], padding[1])
                paddingSecondDim = intArrayOf(padding[2], padding[3])
                paddingThirdDim = intArrayOf(padding[4], padding[5])
            }
            3 -> {
                paddingFirstDim = intArrayOf(padding[0], padding[0])
                paddingSecondDim = intArrayOf(padding[1], padding[1])
                paddingThirdDim = intArrayOf(padding[2], padding[2])
            }
            1 -> {
                paddingFirstDim = intArrayOf(padding[0], padding[0])
                paddingSecondDim = intArrayOf(padding[0], padding[0])
                paddingThirdDim = intArrayOf(padding[0], padding[0])
            }
            else -> throw IllegalArgumentException("Invalid padding argument at layer $name")
        }
        return paddingArraysToInputShape(paddingFirstDim, paddingSecondDim, paddingThirdDim)
    }

    private fun paddingArraysToInputShape(
        paddingFirstDim: IntArray,
        paddingSecondDim: IntArray,
        paddingThirdDim: IntArray
    ): Array<IntArray> {
        return when (inputShape.numDimensions()) {
            5 -> arrayOf(intArrayOf(0, 0), paddingFirstDim, paddingSecondDim, paddingThirdDim, intArrayOf(0, 0))
            4 -> arrayOf(paddingFirstDim, paddingSecondDim, paddingThirdDim, intArrayOf(0, 0))
            3 -> arrayOf(paddingFirstDim, paddingSecondDim, paddingThirdDim)
            else -> throw IllegalArgumentException("Invalid input shape $inputShape")
        }
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "ZeroPadding3D(padding=$padding)"
    }
}
