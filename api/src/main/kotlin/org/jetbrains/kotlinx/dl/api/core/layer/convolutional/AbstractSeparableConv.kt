package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import kotlin.math.roundToInt

/**
 * Abstract N-D convolution layer with separable filters.
 *
 * This layer performs a depthwise convolution that acts separately on
 * channels, followed by a pointwise convolution that mixes channels.
 *
 * If [useBiasInternal] is `true` and a [biasInitializerInternal] is provided,
 * it adds a bias vector to the output.
 * It then optionally applies an activation function to produce the final output.
 *
 * Note: layer attributes cannot be modified after the layer has been called once (except the `trainable` attribute).
 *
 * TODO: add rank for getting the channel axis?
 * TODO: add docs for params?
 * TODO: add trainable param?
 */
public abstract class AbstractSeparableConv(
    protected val filtersInternal: Long,
    protected val kernelSizeInternal: LongArray,
    protected val stridesInternal: LongArray,
    protected val dilationsInternal: LongArray,
    protected val depthMulitplierInternal: Int = 1,
    protected val activationInternal: Activations,
    protected val depthwiseInitializerInternal: Initializer,
    protected val pointwiseInitializerInternal: Initializer,
    protected val biasInitializerInternal: Initializer,
    protected val depthwiseRegularizerInternal: Regularizer?,
    protected val pointwiseRegularizerInternal: Regularizer?,
    protected val biasRegularizerInternal: Regularizer?,
    protected val useBiasInternal: Boolean,
    protected val depthwiseKernelVariableName: String,
    protected val pointwiseKernelVariableName: String,
    protected val biasVariableName: String,
    name: String
) : Layer(
    name
) {

    init {
        require(dilationsInternal.any { it != 1L } && stridesInternal.any { it != 1L }) {
            "Specifying any dilations value != 1 is incompatible with specifying any stride value != 1"
        }
    }

    /** Returns the shape of kernel weights. */
    public val depthwiseShapeArray: LongArray get() = TensorShape(depthwiseKernelShape).dims()

    /** Returns the shape of kernel weights. */
    public val pointwiseShapeArray: LongArray get() = TensorShape(pointwiseKernelShape).dims()

    /** Returns the shape of bias weights. */
    public val biasShapeArray: LongArray get() = TensorShape(biasShape).dims()

    override val hasActivation: Boolean get() = true

    override val paramCount: Int
        get() = (depthwiseKernelShape.numElements() + pointwiseKernelShape.numElements() + biasShape.numElements()).toInt()

    override var weights: Map<String, Array<*>>
        get() = extractDepthConvWeights()
        set(value) = assignWeights(value)

    // weight tensors
    protected lateinit var depthwiseKernel: Variable<Float>
    protected lateinit var pointwiseKernel: Variable<Float>
    protected var bias: Variable<Float>? = null

    // weight tensor shapes
    protected lateinit var depthwiseKernelShape: Shape
    protected lateinit var pointwiseKernelShape: Shape
    protected lateinit var biasShape: Shape

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {

        // Amount of channels should be the last value in the inputShape
        val numberOfChannels = inputShape.size(inputShape.numDimensions() - 1)

        // Compute shapes of kernel and bias matrices
        computeMatricesShapes(numberOfChannels)

        // should be calculated before addWeight because it's used in calculation,
        // need to rewrite addWeight to avoid strange behaviour calculate fanIn, fanOut
        val inputDepth = getInputDepth(numberOfChannels) // number of input channels
        val outputDepth = getOutputDepth(numberOfChannels) // number of output channels

        fanIn = (inputDepth * multiply(*kernelSizeInternal)).toInt()
        fanOut = ((outputDepth * multiply(*kernelSizeInternal)).toDouble() /
                multiply(*stridesInternal).toDouble()).roundToInt()

        val (depthwiseKernelVariableName, pointwiseKernelVariableName, biasVariableName) = defineVariableNames()

        createSeparableConvVariables(
            tf,
            depthwiseKernelVariableName,
            pointwiseKernelVariableName,
            biasVariableName,
            kGraph
        )
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        val shape = defineOutputShape(inputShape)
        outputShape = TensorShape(shape)
        return shape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        var output = separableConvImplementation(tf, input)

        if (useBiasInternal) {
            output = tf.nn.biasAdd(output, bias)
        }

        return Activations.convert(activationInternal).apply(tf, output, name)
    }

    private fun defineVariableNames(): Triple<String, String, String> {
        return if (name.isNotEmpty()) {
            Triple(
                depthwiseKernalVarName(name),
                pointwiseKernelVarName(name),
                biasVarName(name)
            )
        } else {
            Triple(depthwiseKernelVariableName, pointwiseKernelVariableName, biasVariableName)
        }
    }

    private fun createSeparableConvVariables(
        tf: Ops,
        depthwiseKernelVariableName: String,
        pointwiseKernelVariableName: String,
        biasVariableName: String,
        kGraph: KGraph
    ) {
        depthwiseKernel = tf.withName(depthwiseKernelVariableName).variable(depthwiseKernelShape, getDType())
        pointwiseKernel = tf.withName(pointwiseKernelVariableName).variable(pointwiseKernelShape, getDType())
        if (useBiasInternal) bias = tf.withName(biasVariableName).variable(biasShape, getDType())

        depthwiseKernel = addWeight(
            tf,
            kGraph,
            depthwiseKernelVariableName,
            depthwiseKernel,
            depthwiseInitializerInternal,
            depthwiseRegularizerInternal
        )
        pointwiseKernel = addWeight(
            tf,
            kGraph,
            pointwiseKernelVariableName,
            pointwiseKernel,
            pointwiseInitializerInternal,
            pointwiseRegularizerInternal
        )
        if (useBiasInternal)
            bias = addWeight(tf, kGraph, biasVariableName, bias!!, biasInitializerInternal, biasRegularizerInternal)
    }

    protected open fun getInputDepth(numberOfChannels: Long): Long = numberOfChannels

    protected open fun getOutputDepth(numberOfChannels: Long): Long = filtersInternal

    private fun computeMatricesShapes(numberOfChannels: Long) {
        depthwiseKernelShape = shapeFromDims(*kernelSizeInternal, numberOfChannels, depthMulitplierInternal.toLong())
        pointwiseKernelShape = shapeFromDims(1, 1, numberOfChannels * depthMulitplierInternal, filtersInternal)
        biasShape = Shape.make(filtersInternal)
    }

    private fun extractDepthConvWeights(): Map<String, Array<*>> {
        return extractWeights(defineVariableNames().toList())
    }

    protected abstract fun depthwiseKernalVarName(name: String): String

    protected abstract fun pointwiseKernelVarName(name: String): String

    protected abstract fun biasVarName(name: String): String

    protected abstract fun separableConvImplementation(tf: Ops, input: Operand<Float>): Operand<Float>

    protected abstract fun defineOutputShape(inputShape: Shape): Shape
}

private fun multiply(vararg values: Long) = values.fold(1L, Long::times)
