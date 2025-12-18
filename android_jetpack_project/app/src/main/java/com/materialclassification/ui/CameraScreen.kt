package com.materialclassification.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.ui.graphics.Color
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.materialclassification.camera.CameraManager
import com.materialclassification.classification.ClassificationResult
import com.materialclassification.classification.ClassificationApiService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun CameraScreen(
    apiService: ClassificationApiService,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    var hasPermission by remember { mutableStateOf(false) }
    var useSvm by remember { mutableStateOf(false) }
    var apiUrl by remember { mutableStateOf("http://10.0.2.2:8000") }
    var classificationResult by remember { mutableStateOf<ClassificationResult>(ClassificationResult("Waiting...", 0.0f)) }
    var cameraManager by remember { mutableStateOf<CameraManager?>(null) }
    val previewView = remember { PreviewView(context) }
    val classificationScope = remember { CoroutineScope(SupervisorJob() + Dispatchers.Default) }
    
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasPermission = isGranted
        if (!isGranted) {
            Toast.makeText(context, "Camera permission required", Toast.LENGTH_LONG).show()
        }
    }
    
    LaunchedEffect(Unit) {
        hasPermission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        
        if (!hasPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
    
    LaunchedEffect(hasPermission) {
        if (hasPermission && cameraManager == null) {
            cameraManager = CameraManager(
                context = context,
                previewView = previewView,
                lifecycleOwner = lifecycleOwner,
                onFrameCaptured = { bitmap ->
                    android.util.Log.d("CameraScreen", "Frame captured, bitmap: ${bitmap != null}")
                    if (bitmap != null) {
                        classificationScope.launch {
                            try {
                                android.util.Log.d("CameraScreen", "Calling API service")
                                val currentUseSvm = useSvm
                                val currentUrl = apiUrl
                                val result = apiService.classify(bitmap, currentUseSvm, currentUrl)
                                android.util.Log.d("CameraScreen", "Got result: ${result.className}")
                                withContext(Dispatchers.Main) {
                                    classificationResult = result
                                }
                            } catch (e: Exception) {
                                android.util.Log.e("CameraScreen", "Error: ${e.message}", e)
                                e.printStackTrace()
                                withContext(Dispatchers.Main) {
                                    classificationResult = ClassificationResult("Error: ${e.message}", 0.0f)
                                }
                            }
                        }
                    } else {
                        android.util.Log.w("CameraScreen", "Bitmap is null, skipping classification")
                    }
                }
            )
            cameraManager?.startCamera()
        }
    }
    
    DisposableEffect(Unit) {
        onDispose {
            cameraManager?.shutdown()
            cameraManager = null
        }
    }
    
    Box(modifier = modifier.fillMaxSize()) {
        if (hasPermission) {
            AndroidView(
                factory = { previewView },
                modifier = Modifier.fillMaxSize()
            )
            
            Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
            ) {
                TextField(
                    value = apiUrl,
                    onValueChange = { apiUrl = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    label = { Text("API URL") }
                )
                
                Button(
                    onClick = { },
                    modifier = Modifier
                        .align(Alignment.CenterHorizontally)
                        .padding(16.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF4CAF50)
                    )
                ) {
                    Text(classificationResult.className)
                }
                
                Button(
                    onClick = { useSvm = !useSvm },
                    modifier = Modifier
                        .align(Alignment.CenterHorizontally)
                        .padding(16.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF4CAF50)
                    )
                ) {
                    Text(if (useSvm) "Using SVM" else "Using KNN")
                }
            }
        } else {
            Text(
                text = "Requesting camera permission...",
                modifier = Modifier.align(Alignment.Center)
            )
        }
    }
}


