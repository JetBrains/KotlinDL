# KotlinDL: High-level Deep Learning API in Kotlin [![official JetBrains project](http://jb.gg/badges/incubator.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)

[![Kotlin](https://img.shields.io/badge/kotlin-1.4.32-blue.svg?logo=kotlin)](http://kotlinlang.org)
[![Slack channel](https://img.shields.io/badge/chat-slack-green.svg?logo=slack)](https://kotlinlang.slack.com/messages/kotlindl/)

KotlinDL is a high-level Deep Learning API written in Kotlin and inspired by [Keras](https://keras.io). 
Under the hood, it uses TensorFlow Java API. KotlinDL offers simple APIs for training deep learning models from scratch, 
importing existing Keras models for inference, and leveraging transfer learning for tailoring existing pre-trained models to your tasks. 

This project aims to make Deep Learning easier for JVM developers and simplify deploying deep learning models in JVM production environments.
 
Here's an example of what a classic convolutional neural network LeNet would look like in KotlinDL:

```kotlin
private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L
private const val TEST_BATCH_SIZE = 1000

private val lenet5Classic = Sequential.of(
    Input(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS
    ),
    Conv2D(
        filters = 6,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Tanh,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.VALID
    ),
    Conv2D(
        filters = 16,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Tanh,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.VALID
    ),
    Flatten(), // 3136
    Dense(
        outputSize = 120,
        activation = Activations.Tanh,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Constant(0.1f)
    ),
    Dense(
        outputSize = 84,
        activation = Activations.Tanh,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Constant(0.1f)
    ),
    Dense(
        outputSize = NUMBER_OF_CLASSES,
        activation = Activations.Linear,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Constant(0.1f)
    )
)


fun main() {
    val (train, test) = mnist()
    
    lenet5Classic.use {
        it.compile(
            optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
    
        it.summary()
    
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)
    
        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
    
        println("Accuracy: $accuracy")
    }
}
```

## Table of Contents

- [TensorFlow Engine](#tensorflow-engine)
- [Limitations](#limitations)
- [How to configure KotlinDL in your project](#how-to-configure-kotlindl-in-your-project)
- [Working with KotlinDL in Jupyter Notebook](#working-with-kotlindl-in-jupyter-notebook)
- [Examples and tutorials](#examples-and-tutorials)
- [Running KotlinDL on GPU](#running-kotlindl-on-gpu)
- [Logging](#logging)
- [Fat Jar issue](#fat-jar-issue)
- [Reporting issues/Support](#reporting-issuessupport)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

## TensorFlow Engine
KotlinDL is built on top of the TensorFlow 1.15 Java API. 
The Java API for TensorFlow 2.+ has recently had its first public release, and this project will be switching to it in the nearest future. 
This, however, does not affect the high-level API.

## Limitations
Currently, only a limited set of deep learning architectures are supported. Here's the list of available layers:
 
- Input()
- Flatten()
- Dense()
- Dropout()
- Conv2D()
- MaxPool2D()
- AvgPool2D()   
- BatchNorm
- ActivationLayer
- DepthwiseConv2D
- SeparableConv2D
- Merge layers (Add, Subtract, Multiply, Average, Concatenate, Maximum, Minimum)
- GlobalAvgPool2D
- Cropping2D
- Reshape
- ZeroPadding2D 

KotlinDL supports model inference in JVM backend applications. Android support is coming in later releases.  

## How to configure KotlinDL in your project
To use KotlinDL in your project, add the following dependency to your `build.gradle` file:
```kotlin
   repositories {
      mavenCentral()
   }
   
   dependencies {
       implementation 'org.jetbrains.kotlinx:kotlin-deeplearning-api:[KOTLIN-DL-VERSION]'
   }
```
The latest KotlinDL version is 0.2.0. 
The latest stable KotlinDL version is 0.2.0. 

For more details, as well as for `pom.xml` and `build.gradle.kts` examples, please refer to the
[Quick Start Guide](https://nbviewer.jupyter.org/github/avan1235/KotlinDL/blob/notebooks/docs/quick_start_guide.ipynb).

## Working with KotlinDL in Jupyter Notebook
You can work with KotlinDL interactively in Jupyter Notebook with the Kotlin kernel. To do so, add the following dependency in your notebook: 

```
   @file:DependsOn("org.jetbrains.kotlinx:kotlin-deeplearning-api:[KOTLIN-DL-VERSION]")
```

For more details on installing Jupyter Notebook and adding the Kotlin kernel, check out the
[Quick Start Guide](https://nbviewer.jupyter.org/github/avan1235/KotlinDL/blob/notebooks/docs/quick_start_guide.ipynb).

## Examples and tutorials
You do not need to have any prior deep learning experience to start using KotlinDL. 
We are working on including extensive documentation to help you get started. 
At this point, please feel free to check out the following tutorials we have prepared:
- [Quick Start Guide](https://nbviewer.jupyter.org/github/avan1235/KotlinDL/blob/notebooks/docs/quick_start_guide.ipynb) 
- [Creating your first neural network](https://nbviewer.jupyter.org/github/avan1235/KotlinDL/blob/notebooks/docs/create_your_first_nn.ipynb)
- [Training a model](https://nbviewer.jupyter.org/github/avan1235/KotlinDL/blob/notebooks/docs/training_a_model.ipynb)
- [Running inference with a trained model](https://nbviewer.jupyter.org/github/avan1235/KotlinDL/blob/notebooks/docs/loading_trained_model_for_inference.ipynb)
- [Importing a Keras model](https://nbviewer.jupyter.org/github/avan1235/KotlinDL/blob/notebooks/docs/importing_keras_model.ipynb) 
- [Transfer learning](https://nbviewer.jupyter.org/github/avan1235/KotlinDL/blob/notebooks/docs/transfer_learning.ipynb)

For more inspiration, take a look at the [code examples](examples) in this repo.

## Running KotlinDL on GPU

To enable the training and inference on a GPU, please read this [TensorFlow GPU Support page](https://www.tensorflow.org/install/gpu) 
and install the CUDA framework to enable calculations on a GPU device.

Note that only NVIDIA devices are supported.

You will also need to add the following dependencies in your project if you wish to leverage a GPU: 

```
  compile 'org.tensorflow:libtensorflow:1.15.0'_
  compile 'org.tensorflow:libtensorflow_jni_gpu:1.15.0'_
```

On Windows, the following distributions are required:
- CUDA cuda_10.0.130_411.31_win10
- [cudnn-10.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.3.30/Production/10.0_20190822/cudnn-10.0-windows10-x64-v7.6.3.30.zip)
- [C++ redistributable parts](https://www.microsoft.com/en-us/download/details.aspx?id=48145) 

## Logging

By default, the API module uses the [kotlin-logging](https://github.com/MicroUtils/kotlin-logging) library to organize the logging process separately from the specific logger implementation.

You could use any widely known JVM logging library with a [Simple Logging Facade for Java (SLF4J)](http://www.slf4j.org/) implementation such as Logback or Log4j/Log4j2.

You will also need to add the following dependencies and configuration file ``log4j2.xml`` to the ``src/resource`` folder in your project if you wish to use log4j2:

```
  compile 'org.apache.logging.log4j:log4j-api:2.14.0'
  compile 'org.apache.logging.log4j:log4j-core:2.14.0'
  compile 'org.apache.logging.log4j:log4j-slf4j-impl:2.14.0'
```

```
<Configuration status="WARN">
    <Appenders>
        <Console name="STDOUT" target="SYSTEM_OUT">
            <PatternLayout pattern="%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n"/>
        </Console>
    </Appenders>
    <Loggers>
        <Root level="debug">
            <AppenderRef ref="STDOUT" level="DEBUG"/>
        </Root>
    </Loggers>
</Configuration>

```

If you wish to use Logback, include the following dependency and configuration file ``logback.xml`` to ``src/resource`` folder in your project

```
  compile 'ch.qos.logback:logback-classic:1.2.3'
``` 

```
<configuration>
    <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>

    <root level="info">
        <appender-ref ref="STDOUT"/>
    </root>
</configuration>
```
These configuration files can be found in the `examples` module.

## Fat Jar issue

There is a known Stack Overflow [question](https://stackoverflow.com/questions/47477069/issue-running-tensorflow-with-java/52003343) 
and TensorFlow [issue](https://github.com/tensorflow/tensorflow/issues/30488) with Fat Jar creation and execution on Amazon EC2 instances.

```
java.lang.UnsatisfiedLinkError: /tmp/tensorflow_native_libraries-1562914806051-0/libtensorflow_jni.so: libtensorflow_framework.so.1: cannot open shared object file: No such file or directory
```

Despite the fact that the [bug](https://github.com/tensorflow/tensorflow/issues/30488) describing this problem was closed in the release of TensorFlow 1.14, 
it was not fully fixed and required an additional line in the build script.

One simple [solution](https://github.com/tensorflow/tensorflow/issues/30635#issuecomment-615513958) is to add a TensorFlow version specification to the Jar's Manifest. 
Below you can find an example of a Gradle build task for Fat Jar creation.

```
task fatJar(type: Jar) {
    manifest {
        attributes 'Implementation-Version': '1.15'
    }
    classifier = 'all'
    from { configurations.runtimeClasspath.collect { it.isDirectory() ? it : zipTree(it) } }
    with jar
}
```

## Contributing

Read the [Contributing Guidelines](https://github.com/JetBrains/KotlinDL/blob/master/CONTRIBUTING.md).

## Reporting issues/Support

Please use [GitHub issues](https://github.com/JetBrains/KotlinDL/issues) for filing feature requests and bug reports. 
You are also welcome to join the [#kotlindl channel](https://kotlinlang.slack.com/messages/kotlindl/) in the Kotlin Slack.

## Code of Conduct
This project and the corresponding community are governed by the [JetBrains Open Source and Community Code of Conduct](https://confluence.jetbrains.com/display/ALL/JetBrains+Open+Source+and+Community+Code+of+Conduct). Please make sure you read it. 

## License
KotlinDL is licensed under the [Apache 2.0 License](LICENSE).
