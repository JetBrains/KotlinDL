package examples.inference.keras.vgg


import api.inference.keras.buildModelByJSONConfig
import api.inference.keras.loadWeights
import api.keras.dataset.ImageDataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import io.jhdf.HdfFile
import java.io.File

fun main() {
    val jsonConfigFilePath = "C:\\zaleslaw\\home\\models\\vgg19\\modelConfig.json"

    val jsonConfigFile = File(jsonConfigFilePath)

    val model = buildModelByJSONConfig<Float>(jsonConfigFile)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()
        println(it.kGraph)
        val pathToWeights = "C:\\zaleslaw\\home\\models\\vgg19\\hdf5\\weights.h5"
        val file = File(pathToWeights)
        val hdfFile = HdfFile(file)

        it.loadWeights(hdfFile)

        for (i in 1..8) {
            println("image$i")
            val inputStream = ImageDataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = loadImageAndConvertToFloatArray(inputStream)

            // TODO: need to rewrite predict and getactivations method for inference model (predict on image)

            val (res, activations) = it.predictAndGetActivations(floatArray, "Softmax")
            println(res)
            //drawActivations(activations)

            val predictionVector = it.predictSoftly(floatArray, "Softmax").toMutableList()
            val predictionVector2 = it.predictSoftly(floatArray, "Softmax").toMutableList()


            val top5: MutableMap<Int, Int> = mutableMapOf()
            for (j in 1..5) {

                val max = predictionVector2.maxOrNull()
                val indexOfElem = predictionVector.indexOf(max!!)
                top5[j] = indexOfElem
                predictionVector2.remove(max)
            }

            println(top5.toString())
        }


        /*var weights = it.layers[0].getWeights() // first conv2d layer
        println(weights.size)

        drawFilters(weights[0])

        var weights4 = it.layers[4].getWeights() // first conv2d layer
        println(weights4.size)

        drawFilters(weights4[0])*/
    }
}



