package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

public class ZeroPadding1D : Layer {
    public val padding: IntArray
    private lateinit var inputShape: Shape

    public constructor(
        padding: Int,
        name: String = ""
    ) : this(
        IntArray(2) { padding },
        name
    )

    public constructor(
        padding: Pair<Int, Int>,
        name: String = ""
    ) : this(
        intArrayOf(padding.first, padding.second),
        name
    )

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

