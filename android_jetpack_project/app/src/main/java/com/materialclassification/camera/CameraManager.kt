package com.materialclassification.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraManager(
    private val context: Context,
    private val previewView: PreviewView,
    private val lifecycleOwner: LifecycleOwner,
    private val onFrameCaptured: (Bitmap) -> Unit
) {
    private var imageAnalyzer: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private var lastProcessTime = 0L
    private val frameInterval = 500L

    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return

        val preview = androidx.camera.core.Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also {
                android.util.Log.d("CameraManager", "Setting up ImageAnalysis analyzer")
                it.setAnalyzer(executor) { imageProxy ->
                    android.util.Log.d("CameraManager", "Analyzer called with ImageProxy")
                    processImageProxy(imageProxy)
                }
            }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalyzer
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        android.util.Log.d("CameraManager", "Frame received, format: ${imageProxy.format}")
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastProcessTime < frameInterval) {
            imageProxy.close()
            return
        }
        lastProcessTime = currentTime

        try {
            android.util.Log.d("CameraManager", "Processing frame, format: ${imageProxy.format}")
            val bitmap = imageProxy.toBitmap()
            android.util.Log.d("CameraManager", "Bitmap converted: ${bitmap != null}")
            if (bitmap != null) {
                val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)
                android.util.Log.d("CameraManager", "Calling onFrameCaptured")
                onFrameCaptured(rotatedBitmap)
            } else {
                android.util.Log.w("CameraManager", "Bitmap conversion returned null")
            }
        } catch (e: Exception) {
            android.util.Log.e("CameraManager", "Error processing frame: ${e.message}", e)
            e.printStackTrace()
        } finally {
            imageProxy.close()
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    fun shutdown() {
        try {
            imageAnalyzer?.clearAnalyzer()
            cameraProvider?.unbindAll()
            executor.shutdown()
            if (!executor.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS)) {
                executor.shutdownNow()
            }
        } catch (e: Exception) {
            android.util.Log.e("CameraManager", "Error during shutdown: ${e.message}", e)
            executor.shutdownNow()
        }
    }
}

