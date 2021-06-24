/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.config

/**
 * This class enumerates all possible `padding` types, which could be faced in config-files.
 * Those are strings, like "same", "valid", "full" (for example, from `Conv2D` layers),
 * and numeric values/tuples from `ZeroPadding2D`
 */
internal sealed class KerasPadding {
    object Same : KerasPadding()

    object Valid : KerasPadding()

    object Full : KerasPadding()

    class ZeroPadding1D : KerasPadding {
        val padding:IntArray

        constructor(padding: Int) {
            this.padding = IntArray(1) { padding }
        }

        constructor(padding: IntArray){
            this.padding = IntArray(2)
            this.padding[0] = padding[0]
            this.padding[1] = padding[1]
        }
    }

    class ZeroPadding2D : KerasPadding {
        val padding: IntArray

        constructor(padding: Int) {
            this.padding = IntArray(1) { padding }
        }

        constructor(padding: IntArray) {
            when (padding.size) {
                2 -> {
                    this.padding = IntArray(2)
                    this.padding[0] = padding[0]
                    this.padding[1] = padding[1]
                }
                4 -> {
                    this.padding = IntArray(4)
                    this.padding[0] = padding[0]
                    this.padding[1] = padding[1]
                    this.padding[2] = padding[2]
                    this.padding[3] = padding[3]
                }
                else -> {
                    throw UnsupportedOperationException("Padding with size ${padding.size} is not supported!")
                }
            }
        }

        constructor(padding: Array<IntArray>) {
            this.padding = IntArray(4)
            this.padding[0] = padding[0][0]
            this.padding[1] = padding[0][1]
            this.padding[2] = padding[1][0]
            this.padding[3] = padding[1][1]
        }
    }


    class ZeroPadding3D : KerasPadding {
        val padding: IntArray

        constructor(padding: Int) {
            this.padding = IntArray(1) { padding }
        }

        constructor(padding: IntArray) {
            when (padding.size) {
                3 -> {
                    this.padding = IntArray(3)
                    this.padding[0] = padding[0]
                    this.padding[1] = padding[1]
                    this.padding[2] = padding[2]
                }
                6 -> {
                    this.padding = IntArray(6)
                    this.padding[0] = padding[0]
                    this.padding[1] = padding[1]
                    this.padding[2] = padding[2]
                    this.padding[3] = padding[3]
                    this.padding[4] = padding[4]
                    this.padding[5] = padding[5]
                }
                else -> {
                    throw UnsupportedOperationException("Padding with size ${padding.size} is not supported!")
                }
            }
        }

        constructor(padding: Array<IntArray>) {
            when (padding[0].size) {
                3 -> {
                    this.padding = IntArray(6)
                    this.padding[0] = padding[0][0]
                    this.padding[1] = padding[0][1]
                    this.padding[2] = padding[0][2]
                    this.padding[3] = padding[1][0]
                    this.padding[4] = padding[1][1]
                    this.padding[5] = padding[1][2]
                }
                2 -> {
                    this.padding = IntArray(6)
                    this.padding[0] = padding[0][0]
                    this.padding[1] = padding[0][1]
                    this.padding[2] = padding[1][0]
                    this.padding[3] = padding[1][1]
                    this.padding[4] = padding[2][0]
                    this.padding[5] = padding[2][1]
                }
                else -> {
                    throw UnsupportedOperationException("Padding with size ${padding.size} is not supported!")
                }
            }
        }
    }
}
