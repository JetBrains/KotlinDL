/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.toyresnet


import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.fashionMnist

/** Just loading ToyResNet trained in Keras. */
fun main() {
    val (train, test) = fashionMnist()

    val jsonConfigFile = getToyResNetJSONConfigFile()
    val model = Functional.loadModelConfiguration(jsonConfigFile)
    var copiedModel: Functional
    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val hdfFile = getToyResNetWeightsFile()

        it.loadWeights(hdfFile)
        copiedModel = it.copy(copyWeights = true)
        //println(it.kGraph)

        copiedModel.use {
            //println(copiedModel.kGraph)

            copiedModel.layers.forEach { layer ->
                run {
                    val weights = copiedModel.getLayer(layer.name).weights
                    weights.forEach { (varName, arr) ->
                        val assert = arr.contentDeepEquals(it.getLayer(layer.name).weights[varName])

                        println("${layer.name} for $varName weights are equal: $assert")
                    }

                }
            }

            // copiedModel.summary()
            val accuracy = copiedModel.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

            println("Accuracy before: $accuracy")
        }

        val accuracy = it.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

        println("Accuracy before: $accuracy")
    }

}



