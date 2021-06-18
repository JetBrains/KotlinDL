package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Zero-padding layer for 1D input (e.g. audio).
 *
 * This layer can add zeros in the rows of the audio tensor
 *
 * @property [padding] 2 numbers  interpreted as `(left_pad, right_pad)`.
 *
 * @see [Tensorflow implementation](https://github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/layers/convolutional.py#L2699)
 */
public class ZeroPadding1D : Layer {
    public val padding: IntArray
    private lateinit var inputShape: Shape

    /**
     * Constructs an instance of ZeroPadding1D layer
     * @param [padding] symmetric padding applied to width(same on all sides)
     * @param [name] layer name
     */
    public constructor(
        padding: Int,
        name: String = ""
    ) : this(
        IntArray(2) { padding },
        name
    )

    /**
     * Constructs an instance of ZeroPadding1D layer
     * @param [padding] triple of padding values - [padding.first] represents padding on left side
     * and [padding.second] is padding on right side
     * @param [name] layer name
     */
    public constructor(
        padding: Pair<Int, Int>,
        name: String = ""
    ) : this(
        intArrayOf(padding.first, padding.second),
        name
    )

    /**
     * Constructs an instance of ZeroPadding1D layer
     * @param [padding] list of padding values. Size of list must be equal to 2. Those list values maps to
     * the following paddings:
     * padding[0] -> left padding,
     * padding[1] -> right padding,
     * @param [name] layer name
     */
    public constructor(padding: IntArray, name: String = "") : super(name) {
        require(padding.size == 2)
        this.padding = padding
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        this.inputShape = inputShape
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        val length = inputShape.size(1) + padding[0] + padding[1];
        return Shape.make(inputShape.size(1), length, inputShape.size(2))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val pattern = arrayOf(intArrayOf(0, 0), intArrayOf(padding[0], padding[1]), intArrayOf(0, 0))
        val paddingOperand = tf.constant(pattern)
        val constantValue = tf.constant(0f)
        return tf.pad(input, paddingOperand, constantValue)
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "ZeroPadding1D(padding=$padding)"
    }
}

