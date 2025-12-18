package com.materialclassification.classification

import android.graphics.Bitmap
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

@Serializable
data class ClassificationRequest(
    val image: String,
    val classifier: String
)

@Serializable
data class ClassificationResponse(
    val className: String,
    val confidence: Float
)

class ClassificationApiService {
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    
    private val json = Json { ignoreUnknownKeys = true }
    
    suspend fun classify(bitmap: Bitmap, useSvm: Boolean, baseUrl: String): ClassificationResult {
        val imageBytes = bitmapToByteArray(bitmap)
        val classifierType = if (useSvm) "svm" else "knn"
        val response = sendClassificationRequest(imageBytes, classifierType, baseUrl)
        return parseClassificationResponse(response)
    }
    
    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val outputStream = java.io.ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
        return outputStream.toByteArray()
    }
    
    private suspend fun sendClassificationRequest(
        imageBytes: ByteArray,
        classifierType: String,
        baseUrl: String
    ): String {
        return withContext(Dispatchers.IO) {
            val endpoint = "/classify"
            
            val requestBody = buildRequest(imageBytes, classifierType)
            val response = makeHttpRequest(baseUrl + endpoint, requestBody)
            
            response
        }
    }
    
    private fun buildRequest(imageBytes: ByteArray, classifierType: String): String {
        val base64Image = Base64.encodeToString(imageBytes, Base64.NO_WRAP)
        val requestBody = ClassificationRequest(
            image = base64Image,
            classifier = classifierType
        )
        return json.encodeToString(ClassificationRequest.serializer(), requestBody)
    }
    
    private suspend fun makeHttpRequest(url: String, requestBody: String): String {
        return withContext(Dispatchers.IO) {
            val mediaType = "application/json; charset=utf-8".toMediaType()
            val body = requestBody.toRequestBody(mediaType)
            
            val request = Request.Builder()
                .url(url)
                .post(body)
                .addHeader("Content-Type", "application/json")
                .build()
            
            try {
                android.util.Log.d("ClassificationApi", "Sending request to: $url")
                val response = client.newCall(request).execute()
                android.util.Log.d("ClassificationApi", "Response code: ${response.code}")
                if (response.isSuccessful) {
                    val responseBody = response.body?.string()
                    android.util.Log.d("ClassificationApi", "Response body: $responseBody")
                    responseBody ?: throw Exception("Empty response body")
                } else {
                    val errorBody = response.body?.string()
                    android.util.Log.e("ClassificationApi", "HTTP ${response.code}: ${response.message}, Body: $errorBody")
                    throw Exception("HTTP ${response.code}: ${response.message}")
                }
            } catch (e: Exception) {
                android.util.Log.e("ClassificationApi", "Network error: ${e.message}", e)
                throw Exception("Network error: ${e.message}", e)
            }
        }
    }
    
    private fun parseClassificationResponse(response: String): ClassificationResult {
        val apiResponse = json.decodeFromString<ClassificationResponse>(response)
        return ClassificationResult(apiResponse.className, apiResponse.confidence)
    }
}

