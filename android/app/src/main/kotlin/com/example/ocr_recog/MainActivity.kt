package com.example.ocr_recog

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import kotlin.collections.ArrayList
import java.util.Arrays
import android.graphics.RectF
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.google.android.gms.tasks.Tasks
// Import para Coroutines
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint


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
    private val CHANNEL = "com.example.ocr_recog/plate_recognition"
    private var tfliteInterpreter: Interpreter? = null
    private val MODEL_PATH = "license-plate-finetune-v1s_int8.tflite"

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

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "recognizePlate") {
                val imagePath = call.argument<String>("imagePath")

                if (imagePath != null) {
                    // --- ENVOLVE O CÓDIGO COM UMA COROUTINE ---
                    CoroutineScope(Dispatchers.IO).launch {
                        try {
                            if (tfliteInterpreter == null) {
                                loadModelInterpreter()
                            }

                            val originalBitmap = BitmapFactory.decodeFile(imagePath)
                            if (originalBitmap == null) {
                                withContext(Dispatchers.Main) {
                                    result.error("IMAGE_LOAD_FAILED", "Não foi possível carregar a imagem.", null)
                                }
                                return@launch
                            }

                            val inputBuffer = preprocessBitmap(originalBitmap)
                            val outputBuffer = ByteBuffer.allocateDirect(modelOutputBufferLength).order(ByteOrder.nativeOrder())
                            val outputMap = mutableMapOf<Int, Any>()
                            outputMap[0] = outputBuffer // 2. Passe o ByteBuffer para o mapa.

                            tfliteInterpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

                            outputBuffer.rewind()
                            val outputFloatBuffer = outputBuffer.asFloatBuffer()

                            val plateText = postprocessOutput(outputFloatBuffer, originalBitmap)

                            withContext(Dispatchers.Main) {
                                result.success(plateText)
                            }
                        } catch (e: Exception) {
                            Log.e("PlateRecognition", "Erro durante o reconhecimento da placa: ${e.message}", e)
                            withContext(Dispatchers.Main) {
                                result.error("RECOGNITION_ERROR", "Erro no processamento da placa: ${e.message}", e.toString())
                            }
                        }
                    }
                    // ------------------------------------------
                } else {
                    result.error("INVALID_PATH", "Caminho da imagem não fornecido.", null)
                }
            } else {
                result.notImplemented()
            }
        }
    }

    private fun loadModelInterpreter() {
        try {
            val assetFileDescriptor = assets.openFd(MODEL_PATH)
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength

            val options = Interpreter.Options()
            tfliteInterpreter = Interpreter(fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength), options)

            val inputDetails = tfliteInterpreter?.getInputTensor(0)
            val outputDetails = tfliteInterpreter?.getOutputTensor(0)

            Log.d("PlateRecognition", "Modelo TFLite carregado com sucesso.")
            Log.d("PlateRecognition", "Input Shape: ${inputDetails?.shape()?.let { Arrays.toString(it) }}, Input Type: ${inputDetails?.dataType()}")
            Log.d("PlateRecognition", "Output Shape: ${outputDetails?.shape()?.let { Arrays.toString(it) }}, Output Type: ${outputDetails?.dataType()}")
        } catch (e: Exception) {
            Log.e("PlateRecognition", "Erro ao carregar o modelo TFLite: ${e.message}", e)
            throw RuntimeException("Não foi possível carregar o modelo de Machine Learning.", e)
        }
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(modelInputWidth * modelInputHeight * modelInputChannels * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        // Calcula os fatores de escala e padding (letterboxing)
        val scale = Math.min(modelInputWidth.toFloat() / originalWidth, modelInputHeight.toFloat() / originalHeight)
        scaleX = scale
        scaleY = scale

        val scaledWidth = (scale * originalWidth).toInt()
        val scaledHeight = (scale * originalHeight).toInt()

        padX = (modelInputWidth - scaledWidth) / 2.0f
        padY = (modelInputHeight - scaledHeight) / 2.0f

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)

        // Cria um bitmap de fundo (cinza) e desenha a imagem redimensionada sobre ele
        val frameBitmap = Bitmap.createBitmap(modelInputWidth, modelInputHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(frameBitmap)
        canvas.drawColor(android.graphics.Color.GRAY) // Cor de preenchimento
        canvas.drawBitmap(scaledBitmap, padX, padY, null)

        val pixels = IntArray(modelInputWidth * modelInputHeight)
        frameBitmap.getPixels(pixels, 0, modelInputWidth, 0, 0, modelInputWidth, modelInputHeight)

        // Normaliza os pixels e os coloca no buffer
        for (pixelValue in pixels) {
            val r = (pixelValue shr 16 and 0xFF).toFloat()
            val g = (pixelValue shr 8 and 0xFF).toFloat()
            val b = (pixelValue and 0xFF).toFloat()
            inputBuffer.putFloat(r / 255.0f)
            inputBuffer.putFloat(g / 255.0f)
            inputBuffer.putFloat(b / 255.0f)
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    private fun postprocessOutput(outputBuffer: FloatBuffer, originalBitmap: Bitmap): String {
        outputBuffer.rewind()
        val rawOutput = Array(outputNumFeaturesPerBox) { FloatArray(outputNumBoxes) }
        for (featureIdx in 0 until outputNumFeaturesPerBox) {
            for (boxIdx in 0 until outputNumBoxes) {
                rawOutput[featureIdx][boxIdx] = outputBuffer.get()
            }
        }

        val detections = mutableListOf<Detection>()
        val confidenceThreshold = 0.25f
        for (boxIdx in 0 until outputNumBoxes) {
            val confidence = rawOutput[4][boxIdx]
            if (confidence > confidenceThreshold) {
                val xCenter = rawOutput[0][boxIdx]
                val yCenter = rawOutput[1][boxIdx]
                val width = rawOutput[2][boxIdx]
                val height = rawOutput[3][boxIdx]
                detections.add(Detection(xCenter, yCenter, width, height, confidence))
            }
        }
        Log.d("PlateRecognition", "Numero de detecoes de alta confianca antes de NMS: ${detections.size}")

        val iouThreshold = 0.45f
        val finalDetections = applyNms(detections.sortedByDescending { it.confidence }, iouThreshold)

        Log.d("PlateRecognition", "Numero de detecoes apos NMS: ${finalDetections.size}")

        if (finalDetections.isNotEmpty()) {
            val bestDetection = finalDetections.first()
            val plateBox = convertYoloToPixelCoords(bestDetection, originalBitmap)
            Log.d("PlateRecognition", "Placa final detectada com confianca ${bestDetection.confidence}")

            try {
                // --- LÓGICA DE REFINAMENTO DO RECORTE ---
                val boxWidth = plateBox.x2 - plateBox.x1
                val boxHeight = plateBox.y2 - plateBox.y1

                // Define um padding (gordura) para remover as bordas. 10% (0.10f) é um bom começo.
                val horizontalPadding = boxWidth * 0.10f
                val verticalPadding = boxHeight * 0.10f

                // Calcula as novas coordenadas, "encolhendo" a caixa
                val refinedX1 = plateBox.x1 + horizontalPadding
                val refinedY1 = plateBox.y1 + verticalPadding
                val refinedX2 = plateBox.x2 - horizontalPadding
                val refinedY2 = plateBox.y2 - verticalPadding

                // Garante que as coordenadas ainda são válidas
                val x = refinedX1.toInt().coerceIn(0, originalBitmap.width)
                val y = refinedY1.toInt().coerceIn(0, originalBitmap.height)
                val width = (refinedX2 - refinedX1).toInt().coerceAtMost(originalBitmap.width - x).coerceAtLeast(0)
                val height = (refinedY2 - refinedY1).toInt().coerceAtMost(originalBitmap.height - y).coerceAtLeast(0)
                // --- FIM DA LÓGICA DE REFINAMENTO ---

                // Verifique se o bitmap recortado tem um tamanho mínimo para o ML Kit
                if (width >= 32 && height >= 32) {
                    val croppedBitmap = Bitmap.createBitmap(originalBitmap, x, y, width, height)

                    // **PRÓXIMO PASSO:** Vamos pré-processar este bitmap antes de enviar ao OCR
                    val ocrReadyBitmap = preprocessBitmapForOcr(croppedBitmap)
                    val recognizedText = recognizeTextFromBitmap(ocrReadyBitmap) // Usar o bitmap processado

                    val finalPlate = filterAndFormatPlateText(recognizedText)

                    Log.d("PlateRecognition", "Texto cru do OCR: $recognizedText")
                    Log.d("PlateRecognition", "Placa final formatada: $finalPlate")
                    return "Placa detectada: $finalPlate"
                } else {
                    // Se a imagem recortada for muito pequena, retorne um erro específico.
                    Log.e("PlateRecognition", "Erro de OCR: Imagem recortada muito pequena (w=$width, h=$height).")
                    return "Erro de OCR: Imagem recortada muito pequena."
                }
            } catch (e: Exception) {
                Log.e("PlateRecognition", "Erro ao recortar imagem ou no OCR: ${e.message}", e)
                return "Erro ao fazer OCR: ${e.message}"
            }
        }
        return "Nenhuma placa detectada."
    }

    private fun preprocessBitmapForOcr(bitmap: Bitmap): Bitmap{
        val grayBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(grayBitmap)
        val paint = Paint()

        // Configura a matriz de cores para converter para escala de cinza
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)

        // Desenha o bitmap original no canvas com o filtro de escala de cinza
        canvas.drawBitmap(bitmap, 0f, 0f, paint)

        return grayBitmap
    }
    

    private fun convertYoloToPixelCoords(detection: Detection, originalBitmap: Bitmap): BoundingBox {
        // Coordenadas na imagem de entrada do modelo (640x640)
        val boxX = detection.xCenter
        val boxY = detection.yCenter
        val boxWidth = detection.width
        val boxHeight = detection.height

        // Mapeia as coordenadas de volta para a imagem original
        // 1. Remove o padding
        val denormX = (boxX * modelInputWidth) - padX
        val denormY = (boxY * modelInputHeight) - padY

        // 2. Remove a escala
        val originalX = denormX / scaleX
        val originalY = denormY / scaleY
        val originalWidth = (boxWidth * modelInputWidth) / scaleX
        val originalHeight = (boxHeight * modelInputHeight) / scaleY

        // Calcula os pontos x1, y1, x2, y2 na imagem original
        val x1 = originalX - originalWidth / 2
        val y1 = originalY - originalHeight / 2
        val x2 = originalX + originalWidth / 2
        val y2 = originalY + originalHeight / 2

        return BoundingBox(x1, y1, x2, y2, detection.confidence)
    }

    private fun applyNms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sortedDetections = detections.sortedByDescending { it.confidence }.toMutableList()
        val nonMaxSuppressedDetections = mutableListOf<Detection>()

        while (sortedDetections.isNotEmpty()) {
            val bestDetection = sortedDetections.removeAt(0)
            nonMaxSuppressedDetections.add(bestDetection)

            val remainingDetections = sortedDetections.filter {
                calculateIou(bestDetection, it) < iouThreshold
            }
            sortedDetections.clear()
            sortedDetections.addAll(remainingDetections)
        }
        return nonMaxSuppressedDetections
    }

    private fun calculateIou(boxA: Detection, boxB: Detection): Float {
        val x1A = (boxA.xCenter - boxA.width / 2)
        val y1A = (boxA.yCenter - boxA.height / 2)
        val x2A = (boxA.xCenter + boxA.width / 2)
        val y2A = (boxA.yCenter + boxA.height / 2)

        val x1B = (boxB.xCenter - boxB.width / 2)
        val y1B = (boxB.yCenter - boxB.height / 2)
        val x2B = (boxB.xCenter + boxB.width / 2)
        val y2B = (boxB.yCenter + boxB.height / 2)

        val xIntersection = Math.max(x1A, x1B)
        val yIntersection = Math.max(y1A, y1B)
        val xIntersection2 = Math.min(x2A, x2B)
        val yIntersection2 = Math.min(y2A, y2B)

        val intersectionArea = Math.max(0.0f, xIntersection2 - xIntersection) * Math.max(0.0f, yIntersection2 - yIntersection)

        val areaA = boxA.width * boxA.height
        val areaB = boxB.width * boxB.height

        return intersectionArea / (areaA + areaB - intersectionArea)
    }

    private fun recognizeTextFromBitmap(bitmap: Bitmap): String {
        if (bitmap.width <= 0 || bitmap.height <= 0) {
            return "Erro: Bitmap inválido para OCR."
        }
        val image = InputImage.fromBitmap(bitmap, 0)
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        return try {
            val result = Tasks.await(recognizer.process(image))
            result.text.ifEmpty { "Nenhum texto reconhecido." }
        } catch (e: Exception) {
            Log.e("PlateRecognition", "Erro no OCR do ML Kit: ${e.message}", e)
            "Erro no OCR: ${e.message}"
        }
    }

    private fun filterAndFormatPlateText(rawText: String): String {
        // 1. Remover caracteres ambíguos e espaços, mantendo apenas letras e números.
        // Convertendo para maiúsculas para padronização.
        val cleanText = rawText.uppercase().replace("[^A-Z0-9]".toRegex(), "")

        if (cleanText.length < 7) {
            return "Texto muito curto: '$cleanText'"
        }

        // 2. Tenta encontrar o padrão Mercosul (3 letras, 1 número, 1 letra, 2 números)
        val mercosulRegex = "([A-Z]{3})([0-9])([A-Z])([0-9]{2})".toRegex()
        var match = mercosulRegex.find(cleanText)
        if (match != null) {
            // Encontrou um padrão Mercosul completo
            return match.value
        }

        // 3. Tenta encontrar o padrão antigo (3 letras, 4 números)
        val antigoRegex = "([A-Z]{3})([0-9]{4})".toRegex()
        match = antigoRegex.find(cleanText)
        if (match != null) {
            // Encontrou um padrão antigo completo
            return match.value
        }

        // 4. Se nenhum padrão foi encontrado, retorna o texto limpo com um aviso.
        // Pode ser útil para depuração ver o que o OCR retornou.
        return "Padrão não reconhecido: '$cleanText'"
    }
    override fun onDestroy() {
        super.onDestroy()
        tfliteInterpreter?.close()
    }
}