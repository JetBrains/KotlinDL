/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

import org.jetbrains.kotlinx.dl.datasets.image.ColorOrder
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.streams.toList


public class Loading(val dirLocation: String, val imageShape: LongArray, colorMode: ColorOrder) : ImagePreprocessor {
    override fun apply(image: FloatArray): FloatArray {
        TODO("Not yet implemented")
    }

    internal fun fileToImage(file: File): FloatArray {
        TODO("Not yet implemented")
    }

    public fun prepareFileNames(): Array<File> {
        return Files.list(Paths.get(dirLocation))
            .filter { path: Path -> Files.isRegularFile(path) }
            .filter { path: Path -> path.toString().endsWith(".jpg") || path.toString().endsWith(".png") }
            .map { obj: Path -> obj.toFile() }.toList().toTypedArray()
    }
}