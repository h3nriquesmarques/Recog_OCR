package com.example.ocr_recog

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.RectF
import android.os.Environment
import android.util.Log
import androidx.annotation.NonNull
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Arrays
import java.util.Date
import java.util.Locale

import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream

import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.imgproc.Imgproc

data class Detection(
    val xCenter: Float,
    val yCenter: Float,
    val width: Float,
    val height: Float,
    val confidence: Float
)

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float
)

private var scaleX: Float = 1.0f
private var scaleY: Float = 1.0f
private var padX: Float = 0.0f
private var padY: Float = 0.0f

class MainActivity: FlutterActivity() {
    companion object {
        init {
            System.loadLibrary("opencv_java4")
        }
    }


    private val CHANNEL = "com.example.ocr_recog/plate_recognition"
    private var tfliteInterpreter: Interpreter? = null
    private val MODEL_PATH = "best_float32.tflite"

    private val modelInputWidth: Int = 640
    private val modelInputHeight: Int = 640
    private val modelInputChannels: Int = 3
    private val modelInputDataType: DataType = DataType.FLOAT32
    private val modelInputNormalizeMean: Float = 0.0f
    private val modelInputNormalizeStd: Float = 1.0f

    private val outputBatchSize: Int = 1
    private val outputNumFeaturesPerBox: Int = 5
    private val outputNumBoxes: Int = 8400
    private val modelOutputDataType: DataType = DataType.FLOAT32
    private val modelOutputBufferLength: Int = outputBatchSize * outputNumFeaturesPerBox * outputNumBoxes * 4

    private var lastFrameBitmap: Bitmap? = null

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "recognizePlateFromStream" -> {
                    val bytes = call.argument<ByteArray>("bytes")
                    val height = call.argument<Int>("height")
                    val width = call.argument<Int>("width")
                    val rotation = call.argument<Int>("rotation")

                    if (bytes != null && height != null && width != null && rotation != null) {
                        CoroutineScope(Dispatchers.Default).launch {
                            try {
                                val bitmap = yuv420toBitmap(bytes, width, height)
                                if (bitmap != null) {
                                    val finalBitmap = when (rotation) {
                                        0 -> rotateBitmap(bitmap, 90f)
                                        2 -> rotateBitmap(bitmap, -90f)
                                        3 -> rotateBitmap(bitmap, 180f)
                                        else -> bitmap
                                    }

                                    if (isBlurry(finalBitmap)) {
                                        Log.d("PlateRecognition", "Frame descartado por borrão (blur).")
                                        withContext(Dispatchers.Main) { result.success(null) }
                                        return@launch
                                    }

                                    val plateText = processImage(finalBitmap)
                                    withContext(Dispatchers.Main) {
                                        result.success(plateText)
                                    }
                                } else {
                                    withContext(Dispatchers.Main) { result.success(null) }
                                }
                            } catch (e: Exception) {
                                Log.e("RealtimeError", e.message, e)
                                withContext(Dispatchers.Main) { result.success(null) }
                            }
                        }
                    } else {
                        result.error("INVALID_ARGUMENTS", "Argumentos do stream inválidos.", null)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun drawRectOnBitmap(bitmap: Bitmap, rect: RectF, color: Int): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = android.graphics.Canvas(mutableBitmap)
        val paint = Paint().apply {
            this.color = color
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }
        canvas.drawRect(rect,paint)
        return mutableBitmap
    }

    // Gerenciamento do fluxo de inferência no interpretador de tensores
    private fun processImage(image: Bitmap): String {
        if (tfliteInterpreter == null) {
            loadModelInterpreter()
        }
        val inputBuffer = preprocessBitmap(image)
        val outputBuffer = ByteBuffer.allocateDirect(modelOutputBufferLength).order(ByteOrder.nativeOrder())
        val outputMap = mutableMapOf<Int, Any>()
        outputMap[0] = outputBuffer

        tfliteInterpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
        outputBuffer.rewind()
        val outputFloatBuffer = outputBuffer.asFloatBuffer()

        return postprocessOutput(outputFloatBuffer, image)
    }

    private fun yuv420toBitmap(bytes: ByteArray, width: Int, height: Int): Bitmap? {
        if (bytes.isEmpty()) return null
        val out = ByteArrayOutputStream()
        val yuvImage = YuvImage(bytes, ImageFormat.NV21, width, height, null)
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun loadModelInterpreter() {
        try {
            val assetFileDescriptor = assets.openFd(MODEL_PATH)
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val options = Interpreter.Options()
            tfliteInterpreter = Interpreter(fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength), options)
        } catch (e: Exception) {
            throw RuntimeException("Erro ao carregar o modelo TFLite.", e)
        }
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(modelInputWidth * modelInputHeight * modelInputChannels * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val scale = Math.min(modelInputWidth.toFloat() / bitmap.width, modelInputHeight.toFloat() / bitmap.height)
        scaleX = scale
        scaleY = scale
        val scaledWidth = (scale * bitmap.width).toInt()
        val scaledHeight = (scale * bitmap.height).toInt()
        padX = (modelInputWidth - scaledWidth) / 2.0f
        padY = (modelInputHeight - scaledHeight) / 2.0f

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
        val frameBitmap = Bitmap.createBitmap(modelInputWidth, modelInputHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(frameBitmap)
        canvas.drawColor(android.graphics.Color.GRAY)
        canvas.drawBitmap(scaledBitmap, padX, padY, null)

        val pixels = IntArray(modelInputWidth * modelInputHeight)
        frameBitmap.getPixels(pixels, 0, modelInputWidth, 0, 0, modelInputWidth, modelInputHeight)

        for (pixelValue in pixels) {
            inputBuffer.putFloat((pixelValue shr 16 and 0xFF) / 255.0f)
            inputBuffer.putFloat((pixelValue shr 8 and 0xFF) / 255.0f)
            inputBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
        }

        inputBuffer.rewind()
        this.lastFrameBitmap = frameBitmap
        return inputBuffer
    }

    private fun saveBitmapForDebug(bitmap: Bitmap, suffix: String) {
        try {
            val storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val appDir = File(storageDir, "PlateRecognizerDebug")
            if (!appDir.exists()) appDir.mkdirs()
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
            val imageFile = File(appDir, "DEBUG_${timeStamp}_$suffix.jpg")
            val fos = FileOutputStream(imageFile)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
            fos.close()
        } catch (e: Exception) {
            Log.e("Debug", e.message, e)
        }
    }

    private fun postprocessOutput(outputBuffer: FloatBuffer, correctedBitmap: Bitmap): String {
        var resultString = "Nenhuma placa detectada."
        outputBuffer.rewind()
        val rawOutput = Array(outputNumFeaturesPerBox) { FloatArray(outputNumBoxes) }
        for (f in 0 until outputNumFeaturesPerBox) {
            for (b in 0 until outputNumBoxes) rawOutput[f][b] = outputBuffer.get()
        }

        val detections = mutableListOf<Detection>()
        for (i in 0 until outputNumBoxes) {
            if (rawOutput[4][i] > 0.6f) {
                detections.add(Detection(rawOutput[0][i], rawOutput[1][i], rawOutput[2][i], rawOutput[3][i], rawOutput[4][i]))
            }
        }

        val finalDetections = applyNms(detections.sortedByDescending { it.confidence }, 0.45f)
        val filtered = finalDetections.filter { (it.width / it.height) > 1.5f }

        if (filtered.isNotEmpty()) {
            val plateBox = convertYoloToPixelCoords(filtered.first(), correctedBitmap)
            try {
                val drawRect = RectF(plateBox.x1, plateBox.y1, plateBox.x2, plateBox.y2)
                val imageWithBox = drawRectOnBitmap(correctedBitmap, drawRect, android.graphics.Color.RED)
                saveBitmapForDebug(imageWithBox, "00_Yolo_Box.jpg")

                // --- ALTERAÇÃO 2: Recorte Fiel (Sem encolher a imagem) ---
                val refinedX1 = plateBox.x1
                val refinedY1 = plateBox.y1
                val refinedX2 = plateBox.x2
                val refinedY2 = plateBox.y2

                val x = refinedX1.toInt().coerceIn(0, correctedBitmap.width)
                val y = refinedY1.toInt().coerceIn(0, correctedBitmap.height)
                val width = (refinedX2 - refinedX1).toInt().coerceAtMost(correctedBitmap.width - x).coerceAtLeast(0)
                val height = (refinedY2 - refinedY1).toInt().coerceAtMost(correctedBitmap.height - y).coerceAtLeast(0)

                if (width >= 32 && height >= 32) {
                    val croppedBitmap = Bitmap.createBitmap(correctedBitmap, x, y, width, height)
                    saveBitmapForDebug(croppedBitmap, "01_cropped.jpg")

                    val ocrReadyBitmap = preprocessBitmapForOcr(croppedBitmap)
                    val recognizedText = recognizeTextFromBitmap(ocrReadyBitmap)
                    resultString = filterAndFormatPlateText(recognizedText)
                }
            } catch (e: Exception) {
                Log.e("PlateRecognition", "Erro: ${e.message}")
            }
        }
        return resultString
    }

    
    private fun preprocessBitmapForOcr(bitmap: Bitmap): Bitmap {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        val grayMat = Mat()
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)

        val grayBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(grayMat, grayBitmap)
        saveBitmapForDebug(grayBitmap, "02a_Gray.jpg")

        // Adição de Gaussian Blur (3x3) para limpar o sinal
         val blurredMat = Mat()
         if (bitmap.width > 180){
            Imgproc.GaussianBlur(grayMat, blurredMat, org.opencv.core.Size(3.0, 3.0), 0.0)
         } else {
            grayMat.copyTo(blurredMat)
         }
         val blurredBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
         Utils.matToBitmap(blurredMat, blurredBitmap)
         saveBitmapForDebug(blurredBitmap, "02b_Blurred.jpg")
            // Aplicação de Adaptive Thresholding para binarização e fechamento morfológico para melhorar a detecção de caracteres
            val binaryMat = Mat()
            Imgproc.adaptiveThreshold(blurredMat, binaryMat, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 13, 7.0)
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(2.0, 3.0))
            Imgproc.morphologyEx(binaryMat, binaryMat, Imgproc.MORPH_CLOSE, kernel)
        

        val binaryBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(binaryMat, binaryBitmap)
        saveBitmapForDebug(binaryBitmap, "03_OcrInput.jpg")
        return binaryBitmap
    }

    private fun convertYoloToPixelCoords(detection: Detection, correctedBitmap: Bitmap): BoundingBox {
        val denormX = (detection.xCenter * modelInputWidth) - padX
        val denormY = (detection.yCenter * modelInputHeight) - padY
        val originalX = denormX / scaleX
        val originalY = denormY / scaleY
        val originalWidth = (detection.width * modelInputWidth) / scaleX
        val originalHeight = (detection.height * modelInputHeight) / scaleY
        return BoundingBox(originalX - originalWidth / 2, originalY - originalHeight / 2, originalX + originalWidth / 2, originalY + originalHeight / 2, detection.confidence)
    }

        // Função de supressão de não-máximos (NMS) para filtrar caixas delimitadoras 
    private fun applyNms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val results = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            results.add(best)
            sorted.removeAll { calculateIou(best, it) > iouThreshold }
        }
        return results
    }

    private fun calculateIou(boxA: Detection, boxB: Detection): Float {
        val x1A = boxA.xCenter - boxA.width / 2; val x2A = boxA.xCenter + boxA.width / 2
        val y1A = boxA.yCenter - boxA.height / 2; val y2A = boxA.yCenter + boxA.height / 2
        val x1B = boxB.xCenter - boxB.width / 2; val x2B = boxB.xCenter + boxB.width / 2
        val y1B = boxB.yCenter - boxB.height / 2; val y2B = boxB.yCenter + boxB.height / 2
        val interArea = Math.max(0f, Math.min(x2A, x2B) - Math.max(x1A, x1B)) * Math.max(0f, Math.min(y2A, y2B) - Math.max(y1A, y1B))
        return interArea / (boxA.width * boxA.height + boxB.width * boxB.height - interArea)
    }

    // Função para reconhecer texto usando ML Kit
    private fun recognizeTextFromBitmap(bitmap: Bitmap): String {
        if (bitmap.width <= 0 || bitmap.height <= 0) return ""
        val image = InputImage.fromBitmap(bitmap, 0)
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        return try {
            val result = Tasks.await(recognizer.process(image))
            result.text.ifEmpty { "" }
        } catch (e: Exception) { "" }
    }
    // Função para filtrar e formatar o texto da placa
    private fun filterAndFormatPlateText(rawText: String): String {
        val lines = rawText.uppercase().lines()
        val longestLine = lines.map { it.replace("[^A-Z0-9]".toRegex(), "") }.maxByOrNull { it.length } ?: ""
        if (longestLine.length != 7) return "Padrão não reconhecido: '$longestLine'"
        var chars = longestLine.toCharArray()
        // Substituições comuns de caracteres para melhorar a precisão
        for (i in 0..2) {
            if (chars[i] == '0') chars[i] = 'O'
            if (chars[i] == '1') chars[i] = 'I'
            if (chars[i] == '5') chars[i] = 'S'
        }
        val cleanText = String(chars)
        val mercosulRegex = "[A-Z]{3}[0-9][A-Z][0-9]{2}".toRegex()
        val antigoRegex = "[A-Z]{3}[0-9]{4}".toRegex()
        if (cleanText.matches(mercosulRegex)) return cleanText
        if (cleanText.matches(antigoRegex)) return cleanText
        return "Padrão não reconhecido: '$cleanText'"
    }

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(angle) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        tfliteInterpreter?.close()
    }
    //Filtro de validação de borrão (blur) usando Variância Laplaciana
    private fun isBlurry(bitmap: Bitmap, threshold: Double = 100.0): Boolean {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        val grayMat = Mat()
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)
        val laplacianMat = Mat()
        Imgproc.Laplacian(grayMat, laplacianMat, org.opencv.core.CvType.CV_64F)
        val meanStdDev = MatOfDouble()
        org.opencv.core.Core.meanStdDev(laplacianMat, MatOfDouble(), meanStdDev)
        return Math.pow(meanStdDev.get(0, 0)[0], 2.0) < threshold
    }
}