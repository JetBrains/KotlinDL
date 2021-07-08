/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import io.jhdf.HdfFile
import io.jhdf.api.Group
import io.jhdf.dataset.DatasetBase
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.DepthwiseConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.SeparableConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.util.*

private const val KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/kernel:0"
private const val BIAS_DATA_PATH_TEMPLATE = "/%s/%s/bias:0"
private const val GAMMA_DATA_PATH_TEMPLATE = "/%s/%s/gamma:0"
private const val BETA_DATA_PATH_TEMPLATE = "/%s/%s/beta:0"
private const val MOVING_MEAN_DATA_PATH_TEMPLATE = "/%s/%s/moving_mean:0"
private const val MOVING_VARIANCE_DATA_PATH_TEMPLATE = "/%s/%s/moving_variance:0"
private const val DEPTHWISE_KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/depthwise_kernel:0"
private const val POINTWISE_KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/pointwise_kernel:0"
private const val DEPTHWISE_BIAS_DATA_PATH_TEMPLATE = "/%s/%s/depthwise_bias:0"

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Model model.
 */
public fun GraphTrainableModel.loadWeights(
    hdfFile: HdfFile
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }

    when {
        hdfFile.attributes.containsKey("layer_names") -> loadWeightsFromHdf5Group(hdfFile, this, null)
        hdfFile.children.containsKey("model_weights") -> {
            loadWeightsFromHdf5Group((hdfFile as Group).getChild("model_weights") as Group, this, null)
        }
        else -> {
            this.logger.info { "This is unknown path format. Use special method loadWeightsViaPathTemplates() to specify templates to load weights." }
        }
    }

    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for pre-defined list of layers.
 *
 * NOTE: Weights for another layers will not be loaded (should be initialized manually).
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [layerList] List of layers to load weights. Weights for other layers will be initialized by initializer later.
 */
public fun GraphTrainableModel.loadWeights(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }

    when {
        hdfFile.attributes.containsKey("layer_names") -> loadWeightsFromHdf5Group(hdfFile, this, layerList)
        hdfFile.children.containsKey("model_weights") -> {
            loadWeightsFromHdf5Group((hdfFile as Group).getChild("model_weights") as Group, this, layerList)
        }
        else -> {
            this.logger.info { "This is unknown path format. Use special method loadWeightsViaPathTemplates() to specify templates to load weights." }
        }
    }

    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for non-trainable (or frozen) layers only.
 *
 * NOTE: Weights for trainable layers will not be loaded and will be initialized via default initializers.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 */
public fun GraphTrainableModel.loadWeightsForFrozenLayers(
    hdfFile: HdfFile
) {
    val frozenLayers = mutableListOf<Layer>()
    this.layers.forEach {
        if (!it.isTrainable) frozenLayers.add(it)
    }
    this.loadWeights(hdfFile, frozenLayers)
}

private fun loadWeightsFromHdf5Group(group: Group, model: GraphTrainableModel, layerList: MutableList<Layer>?) {
    var originalKerasVersion = 1

    if (group.attributes.containsKey("keras_version") && ((if (group.attributes["keras_version"] != null) group.attributes["keras_version"]?.data else "1") as String).startsWith(
            "2"
        )
    ) {
        originalKerasVersion = 2
    }
    if (originalKerasVersion == 1
    ) {
        throw UnsupportedOperationException(
            "The weights loading from Keras 1.x is not supported by default!" +
                    "\nUse loadWeightsViaPathTemplates() method to make custom loading!"
        )
    }

    if (layerList != null) {
        model.layers.forEach {
            run {
                if (layerList.contains(it)) {
                    fillLayerWeights(it, group, model)
                } else {
                    initLayerWeights(it, model)
                }
            }
        }
    } else {
        model.layers.forEach {
            run {
                fillLayerWeights(it, group, model)
            }
        }
    }
}

private fun fillLayerWeights(
    it: Layer,
    group: Group,
    model: GraphTrainableModel
) {
    when (it::class) {
        Dense::class -> fillDenseVariablesFromKeras(it.name, group, model)
        Conv2D::class -> fillConv2DVariablesFromKeras(
            it.name,
            group,
            model
        )
        DepthwiseConv2D::class -> fillDepthwiseConv2DVariablesFromKeras(
            it.name,
            group,
            model
        )
        SeparableConv2D::class -> fillSeparableConv2DVariablesFromKeras(
            it.name,
            group,
            model
        )
        BatchNorm::class -> fillBatchNormVariablesFromKeras(it.name, group, model)
        else -> println("No weights loading for ${it.name}")
    }
    model.logger.info { " Weights loaded for ${it.name}. ${it.paramCount} parameters are loaded. " }
}

private fun fillConv2DVariablesFromKeras(
    layerName: String,
    group: Group,
    model: GraphTrainableModel
) {
    val availableLayerNames = group.children.map { e -> group.children[e.key]!!.name }.toList()
    val modelLayerNames = model.layers.map { e -> e.name }.toList()
    val layerWeightsNode = group.children[layerName]
    check(layerWeightsNode != null) {
        "Weights for the loaded Conv2D layer $layerName are not found in .h5 file! " +
                "\nh5 weight file contains weights for the following list of layers: $availableLayerNames" +
                "\nDouble-check your loaded configuration which contains layers with the following names: $modelLayerNames."
    }

    val firstLevelGroup: Group = layerWeightsNode as Group
    val nameOfWeightSubGroup = firstLevelGroup.children.keys.first()
    val dataNodes = (firstLevelGroup.children[nameOfWeightSubGroup] as Group).children

    dataNodes.values.map { it as DatasetBase }.forEach {
        val dims = it.dimensions
        val data = it.data
        when (it.name) {
            "kernel:0" -> {
                val kernelVariableName = convKernelVarName(layerName, dim = 2)
                val kernelShape = (model.getLayer(layerName) as Conv2D).kernelShapeArray
                require(
                    kernelShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${kernelShape.contentToString()}" }
                model.fillVariable(kernelVariableName, data)
            }
            "bias:0" -> {
                val biasVariableName = convBiasVarName(layerName, dim = 2)
                val biasShape = (model.getLayer(layerName) as Conv2D).biasShapeArray
                require(
                    biasShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${biasShape.contentToString()}" }
                model.fillVariable(biasVariableName, data)
            }
            else -> {
                throw IllegalArgumentException("Parsing of h5 file for variable with name ${it.name} in layer $layerName is not supported!")
            }
        }

    }
}

// TODO: copy-paste from the Conv2D method
private fun fillDepthwiseConv2DVariablesFromKeras(
    layerName: String,
    group: Group,
    model: GraphTrainableModel
) {
    val availableLayerNames = group.children.map { e -> group.children[e.key]!!.name }.toList()
    val modelLayerNames = model.layers.map { e -> e.name }.toList()
    val layerWeightsNode = group.children[layerName]
    check(layerWeightsNode != null) {
        "Weights for the loaded Conv2D layer $layerName are not found in .h5 file! " +
                "\nh5 weight file contains weights for the following list of layers: $availableLayerNames" +
                "\nDouble-check your loaded configuration which contains layers with the following names: $modelLayerNames."
    }

    val firstLevelGroup: Group = layerWeightsNode as Group
    val nameOfWeightSubGroup = firstLevelGroup.children.keys.first()
    val dataNodes = (firstLevelGroup.children[nameOfWeightSubGroup] as Group).children

    dataNodes.values.map { it as DatasetBase }.forEach {
        val dims = it.dimensions
        val data = it.data
        when (it.name) {
            "depthwise_kernel:0" -> {
                val kernelVariableName = depthwiseConv2dKernelVarName(layerName)
                val kernelShape = (model.getLayer(layerName) as DepthwiseConv2D).kernelShapeArray
                require(
                    kernelShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${kernelShape.contentToString()}" }
                model.fillVariable(kernelVariableName, data)
            }
            "depthwise_bias:0" -> {
                val biasVariableName = depthwiseConv2dBiasVarName(layerName)
                val biasShape = (model.getLayer(layerName) as DepthwiseConv2D).biasShapeArray
                require(
                    biasShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${biasShape.contentToString()}" }
                model.fillVariable(biasVariableName, data)
            }
            else -> {
                throw IllegalArgumentException("Parsing of h5 file for variable with name ${it.name} in layer $layerName is not supported!")
            }
        }

    }
}

private fun fillSeparableConv2DVariablesFromKeras(
    layerName: String,
    group: Group,
    model: GraphTrainableModel
) {
    val availableLayerNames = group.children.map { e -> group.children[e.key]!!.name }.toList()
    val modelLayerNames = model.layers.map { e -> e.name }.toList()
    val layerWeightsNode = group.children[layerName]
    check(layerWeightsNode != null) {
        "Weights for the loaded Conv2D layer $layerName are not found in .h5 file! " +
                "\nh5 weight file contains weights for the following list of layers: $availableLayerNames" +
                "\nDouble-check your loaded configuration which contains layers with the following names: $modelLayerNames."
    }

    val firstLevelGroup: Group = layerWeightsNode as Group
    val nameOfWeightSubGroup = firstLevelGroup.children.keys.first()
    val dataNodes = (firstLevelGroup.children[nameOfWeightSubGroup] as Group).children

    dataNodes.values.map { it as DatasetBase }.forEach {
        val dims = it.dimensions
        val data = it.data
        when (it.name) {
            "depthwise_kernel:0" -> {
                val kernelVariableName = separableConvDepthwiseKernelVarName(layerName, dim = 2)
                val kernelShape = (model.getLayer(layerName) as SeparableConv2D).depthwiseShapeArray
                require(
                    kernelShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${kernelShape.contentToString()}" }
                model.fillVariable(kernelVariableName, data)
            }
            "pointwise_kernel:0" -> {
                val kernelVariableName = separableConvPointwiseKernelVarName(layerName, dim = 2)
                val kernelShape = (model.getLayer(layerName) as SeparableConv2D).pointwiseShapeArray
                require(
                    kernelShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${kernelShape.contentToString()}" }
                model.fillVariable(kernelVariableName, data)
            }
            "depthwise_bias:0" -> {
                val biasVariableName = separableConvBiasVarName(layerName, dim = 2)
                val biasShape = (model.getLayer(layerName) as SeparableConv2D).biasShapeArray
                require(
                    biasShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${biasShape.contentToString()}" }
                model.fillVariable(biasVariableName, data)
            }
            else -> {
                throw IllegalArgumentException("Parsing of h5 file for variable with name ${it.name} in layer $layerName is not supported!")
            }
        }

    }
}

private fun fillDenseVariablesFromKeras(
    layerName: String,
    group: Group,
    model: GraphTrainableModel
) {
    val firstLevelGroup: Group = group.children[layerName] as Group
    val nameOfWeightSubGroup = firstLevelGroup.children.keys.first()
    val dataNodes = (firstLevelGroup.children[nameOfWeightSubGroup] as Group).children

    dataNodes.values.map { it as DatasetBase }.forEach {
        val dims = it.dimensions
        val data = it.data
        when (it.name) {
            "kernel:0" -> {
                val kernelVariableName = denseKernelVarName(layerName)
                val kernelShape = (model.getLayer(layerName) as Dense).kernelShapeArray
                require(
                    kernelShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${kernelShape.contentToString()}" }
                model.fillVariable(kernelVariableName, data)
            }
            "bias:0" -> {
                val biasVariableName = denseBiasVarName(layerName)
                val biasShape = (model.getLayer(layerName) as Dense).biasShapeArray
                require(
                    biasShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${biasShape.contentToString()}" }
                model.fillVariable(biasVariableName, data)
            }
            else -> {
                throw IllegalArgumentException("Parsing of h5 file for variable with name ${it.name} in layer $layerName is not supported!")
            }
        }
    }
}

// TODO: gamma and beta could be misssed due to batchNorm formula https://stackoverflow.com/questions/43813549/restoring-tensorflow-model-cannot-find-gamma-scale-for-batch-norm-layers-in-the
private fun fillBatchNormVariablesFromKeras(
    layerName: String,
    group: Group,
    model: GraphTrainableModel
) {
    val firstLevelGroup: Group = group.children[layerName] as Group
    val nameOfWeightSubGroup = firstLevelGroup.children.keys.first()
    val dataNodes = (firstLevelGroup.children[nameOfWeightSubGroup] as Group).children

    dataNodes.values.map { it as DatasetBase }.forEach {
        val dims = it.dimensions
        val data = it.data
        when (it.name) {
            "gamma:0" -> {
                val gammaVariableName = batchNormGammaVarName(layerName)
                val gammaShape = (model.getLayer(layerName) as BatchNorm).gammaShapeArray!!
                require(
                    gammaShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Gamma shape in loaded data is ${dims.contentToString()}. Should be ${gammaShape.contentToString()}" }
                model.fillVariable(gammaVariableName, data)
            }
            "beta:0" -> {
                val betaVariableName = batchNormBetaVarName(layerName)
                val betaShape = (model.getLayer(layerName) as BatchNorm).betaShapeArray!!
                require(
                    betaShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Beta shape in loaded data is ${dims.contentToString()}. Should be ${betaShape.contentToString()}" }
                model.fillVariable(betaVariableName, data)
            }
            "moving_mean:0" -> {
                val movingMeanVariableName = batchNormMovingMeanVarName(layerName)
                val movingMeanShape = (model.getLayer(layerName) as BatchNorm).movingMeanShapeArray
                require(
                    movingMeanShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Moving mean shape in loaded data is ${dims.contentToString()}. Should be ${movingMeanShape.contentToString()}" }
                model.fillVariable(movingMeanVariableName, data)
            }
            "moving_variance:0" -> {
                val movingVarianceVariableName = batchNormMovingVarianceVarName(layerName)
                val movingVarianceShape = (model.getLayer(layerName) as BatchNorm).movingVarianceShapeArray
                require(
                    movingVarianceShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Moving variance shape in loaded data is ${dims.contentToString()}. Should be ${movingVarianceShape.contentToString()}" }
                model.fillVariable(movingVarianceVariableName, data)
            }
            else -> {
                throw IllegalArgumentException("Parsing of h5 file for variable with name ${it.name} in layer $layerName is not supported!")
            }
        }

    }
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun GraphTrainableModel.loadWeightsByPathTemplates(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE, // TODO: doesnt' work for batchnorm/depthwise
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.debug { "Starting weights loading.." }
    this.layers.forEach {
        run {
            fillLayerWeights(
                it,
                hdfFile,
                LayerConvOrDensePaths("", kernelDataPathTemplate, biasDataPathTemplate),
                this
            ) // TODO: doesnt' work for batchnorm/depthwise
        }
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for pre-defined list of layers.
 *
 * NOTE: Weights for another layers will not be loaded (should be initialized manually).
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [layerList] List of layers to load weights. Weights for other layers will be initialized by initializer later.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun GraphTrainableModel.loadWeightsByPathTemplates(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }
    this.layers.forEach {
        run {
            if (layerList.contains(it)) {
                fillLayerWeights(
                    it,
                    hdfFile,
                    LayerConvOrDensePaths("", kernelDataPathTemplate, biasDataPathTemplate),
                    this
                ) // TODO: doesnt' work for batchnorm/depthwise
            } else {
                initLayerWeights(it, this)
            }
        }
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

private fun fillLayerWeights(
    it: Layer,
    hdfFile: HdfFile,
    layerPaths: LayerPaths?,
    model: GraphTrainableModel
) {
    when (it::class) {
        Dense::class -> fillDenseVariables(
            it.name,
            hdfFile,
            model,
            (it as Dense).useBias,
            layerPaths
        )
        Conv2D::class -> fillConv2DVariables(
            it.name,
            hdfFile,
            model,
            (it as Conv2D).useBias,
            layerPaths
        )
        DepthwiseConv2D::class -> fillDepthwiseConv2DVariables(
            it.name,
            hdfFile,
            model,
            (it as DepthwiseConv2D).useBias,
            layerPaths
        )
        SeparableConv2D::class -> fillSeparableConv2DVariables(
            it.name,
            hdfFile,
            model,
            (it as SeparableConv2D).useBias,
            layerPaths
        )
        BatchNorm::class -> fillBatchNormVariables(
            it.name,
            hdfFile,
            model,
            layerPaths
        )
        else -> println("No weights loading for ${it.name}")
    }
    model.logger.info { " Weights loaded for ${it.name}. ${it.paramCount} parameters are loaded. " }
}


/*private fun fillLayerWeights(
    it: Layer,
    hdfFile: HdfFile,
    kernelDataPathTemplate: String,
    biasDataPathTemplate: String,
    model: Sequential
) {
    when (it::class) {
        Dense::class -> fillDenseVariables(
            it.name,
            hdfFile,
            model,
            kernelDataPathTemplate,
            biasDataPathTemplate
        )
        Conv2D::class -> fillConv2DVariables(
            it.name,
            hdfFile,
            model,
            kernelDataPathTemplate,
            biasDataPathTemplate
        )
        else -> println("No weights loading for ${it.name}")
    }
    model.logger.info { " Weights loaded for ${it.name}. ${it.paramCount} parameters are loaded. " }
}*/

private fun initLayerWeights(it: Layer, model: GraphTrainableModel) {
    when (it::class) {
        Dense::class -> initDenseVariablesByDefaultInitializer(it.name, model)
        Conv2D::class -> initConv2DVariablesByDefaultInitializer(it.name, model)
        DepthwiseConv2D::class -> initDepthwiseConv2DVariablesByDefaultInitializer(it.name, model)
        SeparableConv2D::class -> initSeparableConv2DVariablesByDefaultInitializer(it.name, model)
        BatchNorm::class -> initBatchNormVariablesByDefaultInitializer(it.name, model)
        else -> println("No default initialization handled for ${it.name}")
    }
    model.logger.info { " Weights initialized for ${it.name}. ${it.paramCount} parameters are initialized. " }
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for non-trainable (or frozen) layers only.
 *
 * NOTE: Weights for trainable layers will not be loaded and will be initialized via default initializers.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun Functional.loadWeightsForFrozenLayersByPathTemplates(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    val frozenLayers = mutableListOf<Layer>()
    this.layers.forEach {
        if (!it.isTrainable) frozenLayers.add(it)
    }
    this.loadWeightsByPathTemplates(hdfFile, frozenLayers, kernelDataPathTemplate, biasDataPathTemplate)
}

private fun initConv2DVariablesByDefaultInitializer(name: String, model: GraphTrainableModel) {
    val kernelVariableName = convKernelVarName(name, dim = 2)
    val biasVariableName = convBiasVarName(name, dim = 2)
    model.runAssignOpByVarName(kernelVariableName)
    model.runAssignOpByVarName(biasVariableName)
}

private fun initDepthwiseConv2DVariablesByDefaultInitializer(name: String, model: GraphTrainableModel) {
    val kernelVariableName = depthwiseConv2dKernelVarName(name)
    val biasVariableName = depthwiseConv2dBiasVarName(name)
    model.runAssignOpByVarName(kernelVariableName)
    model.runAssignOpByVarName(biasVariableName)
}

private fun initSeparableConv2DVariablesByDefaultInitializer(name: String, model: GraphTrainableModel) {
    val depthwiseKernelVariableName = separableConvDepthwiseKernelVarName(name, dim = 2)
    val pointwiseKernelVariableName = separableConvPointwiseKernelVarName(name, dim = 2)
    val biasVariableName = separableConvBiasVarName(name, dim = 2)
    model.runAssignOpByVarName(depthwiseKernelVariableName)
    model.runAssignOpByVarName(pointwiseKernelVariableName)
    model.runAssignOpByVarName(biasVariableName)
}

private fun initBatchNormVariablesByDefaultInitializer(name: String, model: GraphTrainableModel) {
    val betaVariableName = batchNormBetaVarName(name)
    val gammaVariableName = batchNormGammaVarName(name)
    val movingMeanVariableName = batchNormMovingMeanVarName(name)
    val movingVarianceVariableName = batchNormMovingVarianceVarName(name)

    model.runAssignOpByVarName(betaVariableName)
    model.runAssignOpByVarName(gammaVariableName)
    model.runAssignOpByVarName(movingMeanVariableName)
    model.runAssignOpByVarName(movingVarianceVariableName)
}

private fun initDenseVariablesByDefaultInitializer(name: String, model: GraphTrainableModel) {
    val kernelVariableName = denseKernelVarName(name)
    val biasVariableName = denseBiasVarName(name)
    model.runAssignOpByVarName(kernelVariableName)
    model.runAssignOpByVarName(biasVariableName)
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [weightPaths] Fully-specified paths to kernel and bias weights of each layer.
 *
 * NOTE: Kernel and bias will be initialized by default initializers if they are missed in [weightPaths] object.
 */
public fun GraphTrainableModel.loadWeightsByPaths(
    hdfFile: HdfFile,
    weightPaths: List<LayerPaths>,
    missedWeights: MissedWeightsStrategy = MissedWeightsStrategy.INITIALIZE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.debug { "Starting weights loading.." }
    this.layers.forEach {
        run {
            val initializedLayerName = it.name
            val layerWeightPaths = weightPaths.find { initializedLayerName == it.layerName }
            if (layerWeightPaths != null) {
                fillLayerWeights(it, hdfFile, layerWeightPaths, this)
            } else {
                if (missedWeights == MissedWeightsStrategy.LOAD_NEW_FORMAT) {
                    fillLayerWeights(
                        it,
                        hdfFile,
                        null, // TODO: refactor = it doesn't work for batchnorm or depthwise
                        this
                    )
                } else {
                    this.logger.info { "Layer weight paths for ${it.name} are not found in 'weightPaths' object. It will be initialized by default initializer." }
                    initLayerWeights(it, this)
                }
            }
        }
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

// TODO: Refactor and add documentation
public enum class MissedWeightsStrategy {
    INITIALIZE,
    LOAD_NEW_FORMAT
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for pre-defined list of layers.
 *
 * NOTE: Weights for another layers will not be loaded (should be initialized manually).
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [layerList] List of layers to load weights. Weights for other layers will be initialized by initializer later.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun GraphTrainableModel.loadWeightsByPaths(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }
    this.layers.forEach {
        run {
            if (layerList.contains(it)) {
                fillLayerWeights(
                    it,
                    hdfFile,
                    LayerConvOrDensePaths("", kernelDataPathTemplate, biasDataPathTemplate),
                    this
                ) // TODO: does not work for BatchNorm/Depthwise
            } else {
                initLayerWeights(it, this)
            }
        }
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

private fun fillConv2DVariables(
    name: String,
    hdfFile: HdfFile,
    model: GraphTrainableModel,
    useBias: Boolean,
    layerPaths: LayerPaths?
) {
    val kernelDataPathTemplate: String
    val biasDataPathTemplate: String

    if (layerPaths == null) {
        kernelDataPathTemplate = KERNEL_DATA_PATH_TEMPLATE
        biasDataPathTemplate = BIAS_DATA_PATH_TEMPLATE
    } else {
        layerPaths as LayerConvOrDensePaths
        kernelDataPathTemplate = layerPaths.kernelPath
        biasDataPathTemplate = layerPaths.biasPath
    }

    val kernelData = hdfFile.getDatasetByPath(kernelDataPathTemplate.format(name, name)).data
    val kernelVariableName = convKernelVarName(name, dim = 2)
    model.fillVariable(kernelVariableName, kernelData)

    if (useBias) {
        val biasData = hdfFile.getDatasetByPath(biasDataPathTemplate.format(name, name)).data
        val biasVariableName = convBiasVarName(name, dim = 2)
        model.fillVariable(biasVariableName, biasData)
    }
}

private fun fillDepthwiseConv2DVariables(
    name: String,
    hdfFile: HdfFile,
    model: GraphTrainableModel,
    useBias: Boolean,
    layerPaths: LayerPaths?
) {
    val kernelDataPathTemplate: String
    val biasDataPathTemplate: String

    if (layerPaths == null) {
        kernelDataPathTemplate = DEPTHWISE_KERNEL_DATA_PATH_TEMPLATE
        biasDataPathTemplate = DEPTHWISE_BIAS_DATA_PATH_TEMPLATE
    } else {
        layerPaths as LayerConvOrDensePaths
        kernelDataPathTemplate = layerPaths.kernelPath
        biasDataPathTemplate = layerPaths.biasPath
    }

    layerPaths as LayerConvOrDensePaths
    val kernelData = hdfFile.getDatasetByPath(kernelDataPathTemplate.format(name, name)).data
    val kernelVariableName = depthwiseConv2dKernelVarName(name)
    model.fillVariable(kernelVariableName, kernelData)

    if (useBias) {
        val biasData = hdfFile.getDatasetByPath(biasDataPathTemplate.format(name, name)).data
        val biasVariableName = depthwiseConv2dBiasVarName(name)
        model.fillVariable(biasVariableName, biasData)
    }
}

private fun fillSeparableConv2DVariables(
    name: String,
    hdfFile: HdfFile,
    model: GraphTrainableModel,
    useBias: Boolean,
    layerPaths: LayerPaths?
) {
    val depthwiseKernelDataPathTemplate: String
    val pointwiseKernelDataPathTemplate: String
    val biasDataPathTemplate: String

    if (layerPaths == null) {
        depthwiseKernelDataPathTemplate = DEPTHWISE_KERNEL_DATA_PATH_TEMPLATE
        pointwiseKernelDataPathTemplate = POINTWISE_KERNEL_DATA_PATH_TEMPLATE
        biasDataPathTemplate = DEPTHWISE_BIAS_DATA_PATH_TEMPLATE
    } else {
        layerPaths as LayerSeparableConv2DPaths
        depthwiseKernelDataPathTemplate = layerPaths.depthwiseKernelPath
        pointwiseKernelDataPathTemplate = layerPaths.depthwiseKernelPath
        biasDataPathTemplate = layerPaths.biasPath
    }

    layerPaths as LayerConvOrDensePaths
    val depthwiseKernelData = hdfFile.getDatasetByPath(depthwiseKernelDataPathTemplate.format(name, name)).data
    val depthwiseKernelVariableName = separableConvDepthwiseKernelVarName(name, dim = 2)
    model.fillVariable(depthwiseKernelVariableName, depthwiseKernelData)

    val pointwiseKernelData = hdfFile.getDatasetByPath(pointwiseKernelDataPathTemplate.format(name, name)).data
    val pointwiseKernelVariableName = separableConvPointwiseKernelVarName(name, dim = 2)
    model.fillVariable(pointwiseKernelVariableName, pointwiseKernelData)

    if (useBias) {
        val biasData = hdfFile.getDatasetByPath(biasDataPathTemplate.format(name, name)).data
        val biasVariableName = depthwiseConv2dBiasVarName(name)
        model.fillVariable(biasVariableName, biasData)
    }
}

private fun fillBatchNormVariables(
    name: String,
    hdfFile: HdfFile,
    model: GraphTrainableModel,
    layerPaths: LayerPaths?
) {
    val gammaDataPathTemplate: String
    val betaDataPathTemplate: String
    val movingMeanDataPathTemplate: String
    val movingVarianceDataPathTemplate: String

    if (layerPaths == null) {
        gammaDataPathTemplate = GAMMA_DATA_PATH_TEMPLATE
        betaDataPathTemplate = BETA_DATA_PATH_TEMPLATE
        movingMeanDataPathTemplate = MOVING_MEAN_DATA_PATH_TEMPLATE
        movingVarianceDataPathTemplate = MOVING_VARIANCE_DATA_PATH_TEMPLATE
    } else {
        layerPaths as LayerBatchNormPaths
        gammaDataPathTemplate = layerPaths.gammaPath
        betaDataPathTemplate = layerPaths.betaPath
        movingMeanDataPathTemplate = layerPaths.movingMeanPath
        movingVarianceDataPathTemplate = layerPaths.movingVariancePath
    }

    val gammaData = hdfFile.getDatasetByPath(gammaDataPathTemplate.format(name, name)).data
    val betaData = hdfFile.getDatasetByPath(betaDataPathTemplate.format(name, name)).data
    val movingMeanData = hdfFile.getDatasetByPath(movingMeanDataPathTemplate.format(name, name)).data
    val movingVarianceData = hdfFile.getDatasetByPath(movingVarianceDataPathTemplate.format(name, name)).data

    val gammaVariableName = batchNormGammaVarName(name)
    val betaVariableName = batchNormBetaVarName(name)
    val movingMeanVariableName = batchNormMovingMeanVarName(name)
    val movingVarianceVariableName = batchNormMovingVarianceVarName(name)

    model.fillVariable(gammaVariableName, gammaData)
    model.fillVariable(betaVariableName, betaData)
    model.fillVariable(movingMeanVariableName, movingMeanData)
    model.fillVariable(movingVarianceVariableName, movingVarianceData)
}

private fun fillDenseVariables(
    name: String,
    hdfFile: HdfFile,
    model: GraphTrainableModel,
    useBias: Boolean,
    layerPaths: LayerPaths?
) {
    val kernelDataPathTemplate: String
    val biasDataPathTemplate: String

    if (layerPaths == null) {
        kernelDataPathTemplate = KERNEL_DATA_PATH_TEMPLATE
        biasDataPathTemplate = BIAS_DATA_PATH_TEMPLATE
    } else {
        layerPaths as LayerConvOrDensePaths
        kernelDataPathTemplate = layerPaths.kernelPath
        biasDataPathTemplate = layerPaths.biasPath
    }

    val kernelData = hdfFile.getDatasetByPath(kernelDataPathTemplate.format(name, name)).data
    val kernelVariableName = denseKernelVarName(name)
    model.fillVariable(kernelVariableName, kernelData)

    if (useBias) {
        val biasData = hdfFile.getDatasetByPath(biasDataPathTemplate.format(name, name)).data
        val biasVariableName = denseBiasVarName(name)
        model.fillVariable(biasVariableName, biasData)
    }
}

/** Parent class for specific paths to layers in h5 file. Contains only [layerName] field */
public open class LayerPaths(public val layerName: String)

/**
 * Contains [layerName], [kernelPath], [biasPath] for specific layer, found in hdf5 file via
 * ```
 * recursivePrintGroupInHDF5File()
 * ```
 * function call.
 */
public class LayerConvOrDensePaths(
    layerName: String,
    /** */
    public val kernelPath: String,
    /** */
    public val biasPath: String
) : LayerPaths(layerName)

/**
 * Contains [layerName], [depthwiseKernelPath],  [pointwiseKernelPath], [biasPath] for [SeparableConv2D] layer, found in hdf5 file via
 * ```
 * recursivePrintGroupInHDF5File()
 * ```
 * function call.
 */
public class LayerSeparableConv2DPaths(
    layerName: String,
    /** */
    public val depthwiseKernelPath: String,
    /** */
    public val pointwiseKernelPath: String,
    /** */
    public val biasPath: String
) : LayerPaths(layerName)

/**
 * Contains [layerName], [gammaPath],  [betaPath], [movingMeanPath], [movingVariancePath] for [BatchNorm] layer, found in hdf5 file via
 * ```
 * recursivePrintGroupInHDF5File()
 * ```
 * function call.
 */
public class LayerBatchNormPaths(
    layerName: String,
    /** */
    public val gammaPath: String,
    /** */
    public val betaPath: String,
    /** */
    public val movingMeanPath: String,
    /** */
    public val movingVariancePath: String
) : LayerPaths(layerName)
