/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.TrainableModel
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

/**
 * Base abstract class for all layers.
 *
 * @param [name] Layer name. Would be changed if empty during model compilation.
 */
public abstract class Layer(public var name: String) {
    /**
     * True, if layer's weights could be changed during training.
     * If false, layer's weights are frozen and could be changed during the training.
     */
    public var isTrainable: Boolean = true
        set(value) {
            if (value && this is NoGradients)
                throw IllegalStateException("${javaClass.name} can not set `isTrainable` to `true`.")
            field = value
        }

    /** Output data tensor shape. */
    public lateinit var outputShape: TensorShape

    /** Model where this layer is used. */
    public var parentModel: TrainableModel? = null

    /** Returns number of input parameters. */
    protected var fanIn: Int = Int.MIN_VALUE

    /** Returns number of output parameters. */
    protected var fanOut: Int = Int.MIN_VALUE

    /** Returns inbound layers. */
    public var inboundLayers: MutableList<Layer> = mutableListOf()

    /** Returns outbound layers. */
    public var outboundLayers: MutableList<Layer> = mutableListOf()

    /**
     * Extend this function to define variables in layer.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [kGraph] [KGraph] to update it.
     * @param [inputShape] Input shape, result of [computeOutputShape] call from previous layer.
     */
    public abstract fun build(tf: Ops, kGraph: KGraph, inputShape: Shape)


    /**
     * Extend this function to define variables in layer.
     *
     * NOTE: This function should be overridden for layers with multiple inputs.
     * NOTE: Used in Functional API
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [kGraph] [KGraph] to update it.
     */
    public fun buildFromInboundLayers(tf: Ops, kGraph: KGraph) {
        require(inboundLayers.isNotEmpty()) { "There is no inbound layers to compute output shape" }
        build(tf, kGraph, inboundLayers[0].outputShape.toShape())
    }

    /**
     * Computes output shape, based on [inputShape] and [Layer] type.
     */
    public abstract fun computeOutputShape(inputShape: Shape): Shape

    /**
     * Computes output shape, based on input shapes of inbound layers.
     *
     * NOTE: This function should be overridden for layers with multiple inputs.
     * NOTE: Used in Functional API
     */
    public open fun computeOutputShapeFromInboundLayers(): TensorShape {
        require(inboundLayers.isNotEmpty()) { "There is no inbound layers to compute output shape" }
        return TensorShape(computeOutputShape(inboundLayers[0].outputShape.toShape()))
    }

    /**
     * Builds main layer input transformation with [tf]. Depends on [Layer] type.
     */
    public abstract fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float>

    /**
     * Builds main layer input transformation with [tf]. Depends on [Layer] type.
     */
    public open fun forward(
        tf: Ops,
        input: List<Operand<Float>>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return forward(tf, input[0], isTraining, numberOfLosses)
    }

    /**
     * Adds a new weight tensor to the layer
     *
     * @param name     variable name
     * @param variable variable to add
     * @return the created variable.
     */
    protected fun addWeight(
        tf: Ops,
        kGraph: KGraph,
        name: String,
        variable: Variable<Float>,
        initializer: Initializer,
        regularizer: Regularizer? = null
    ): Variable<Float> {
        // require(fanIn != Int.MIN_VALUE) { "fanIn should be calculated before initialization for variable $name" }
        // require(fanOut != Int.MIN_VALUE) { "fanOut should be calculated before initialization for variable $name" }

        val initOp = initializer.apply(fanIn, fanOut, tf, variable, name)
        kGraph.addLayerVariable(variable, isTrainable)
        kGraph.addInitializer(name, initOp)
        if (regularizer != null) kGraph.addVariableRegularizer(variable, regularizer)
        return variable
    }

    /** Important part of functional API. It takes [layers] as input and saves them to the [inboundLayers] of the given layer. */
    public operator fun invoke(vararg layers: Layer): Layer {
        inboundLayers = layers.toMutableList()
        return this
    }

    /** Extract weights values by variable names. */
    protected fun extractWeights(variableNames: List<String>): Map<String, Array<*>> {
        require(parentModel != null) { "Layer $name is not related to any model!" }

        val session = parentModel!!.session
        val runner = session.runner()

        for (variableName in variableNames) {
            runner.fetch(variableName)
        }
        val weights = runner.run().map { it.convertTensorToMultiDimArray() }
        return variableNames.zip(weights).toMap()
    }

    /** Extract weights values by variable names. */
    protected fun assignWeights(weights: Map<String, Array<*>>) {
        require(parentModel != null) { "Layer $name is not related to any model!" }

        for (variableName in weights.keys) {
            parentModel!!.assignVariable(variableName, weights[variableName]!!)
        }
    }

    /** Layer's weights. */
    public abstract var weights: Map<String, Array<*>>

    /** Returns True, if layer has internal activation function. */
    public abstract val hasActivation: Boolean

    /** Returns amount of neurons. */
    public abstract val paramCount: Int
}

internal fun requireArraySize(array: LongArray, size: Int, name: String) =
    require (array.size == size) {
        "$name is expected to have size equal $size but got ${array.size}"
    }
