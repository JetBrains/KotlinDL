/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.core

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Applies an activation function to an output.
 *
 * @property [activation] Activation function.
 * @property [name] Custom layer name.
 * @constructor Creates [Dense] object.
 */
public class ActivationLayer(
    public val activation: Activations = Activations.Relu,
    name: String = ""
) : Layer(name) {

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        //left empty
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        this.outputShape = TensorShape(inputShape)
        return inputShape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return Activations.convert(activation).apply(tf, input, name)
    }

    override val weights: List<Array<*>> get() = emptyList()

    override val hasActivation: Boolean get() = true

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "ActivationLayer(activation=$activation)"
    }
}
