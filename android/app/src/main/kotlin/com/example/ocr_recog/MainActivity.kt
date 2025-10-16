package com.example.ocr_recog

// Import para Coroutines

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
                    val rotation = call.argument<Int>("rotation") // Recebendo o índice da rotação

                    if (bytes != null && height != null && width != null && rotation != null) {
                        CoroutineScope(Dispatchers.Default).launch {
                            try {
                                val bitmap = yuv420toBitmap(bytes, width, height)
                                if (bitmap != null) {
                                    // A lógica de rotação baseada na informação do Flutter
                                    // 0=Retrato(Pé), 1=Paisagem(Esquerda), 2=Retrato(Invertido), 3=Paisagem(Direita)
                                    // O sensor da câmera é naturalmente "paisagem", então precisamos girar no modo retrato.
                                    val finalBitmap = when (rotation) {
                                        0 -> rotateBitmap(bitmap, 90f)
                                        2 -> rotateBitmap(bitmap, -90f)
                                        3 -> rotateBitmap(bitmap, 180f)
                                        else -> bitmap // Rotação 1 (LandscapeLeft) já está correta
                                    }

                                    if (isBlurry(finalBitmap)) {
                                        Log.d("PlateRecognition", "Frame descartado por borrão (blur).")
                                        withContext(Dispatchers.Main) { result.success(null) } // Retorna nulo e espera o próximo frame
                                        return@launch // Para a execução deste frame
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
                "recognizePlate" -> {
                    // ... seu código de captura única ...
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun processImage(image: Bitmap): String {
        // Garante que o interpretador TFLite está carregado
        if (tfliteInterpreter == null) {
            loadModelInterpreter()
        }

        // O pipeline que já conhecemos:
        // 1. Pré-processa a imagem para o formato que o YOLO espera
        val inputBuffer = preprocessBitmap(image)

        // 2. Prepara o buffer de saída
        val outputBuffer = ByteBuffer.allocateDirect(modelOutputBufferLength).order(ByteOrder.nativeOrder())
        val outputMap = mutableMapOf<Int, Any>()
        outputMap[0] = outputBuffer

        // 3. Executa a inferência do modelo
        tfliteInterpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        // 4. Prepara o buffer de saída para leitura
        outputBuffer.rewind()
        val outputFloatBuffer = outputBuffer.asFloatBuffer()

        // 5. Executa o pós-processamento para extrair e formatar a placa
        return postprocessOutput(outputFloatBuffer, image)
    }

    private fun yuv420toBitmap(bytes: ByteArray, width: Int, height: Int): Bitmap? {
        if (bytes.isEmpty()) return null
        val out = ByteArrayOutputStream()
        // O formato da câmera do Flutter YUV_420_888 é compatível com o NV21 do Android
        val yuvImage = YuvImage(bytes, ImageFormat.NV21, width, height, null)
        // Comprimimos para JPEG em memória e depois decodificamos para Bitmap
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
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
        this.lastFrameBitmap = frameBitmap
        return inputBuffer
    }

    private fun saveBitmapForDebug(bitmap: Bitmap, suffix: String) {
        try {
            // Define o diretório público de Imagens
            val storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)

            // Cria um subdiretório para o seu app, se não existir
            val appDir = File(storageDir, "PlateRecognizerDebug")
            if (!appDir.exists()) {
                appDir.mkdirs()
            }

            // Cria um nome de arquivo único com data e hora
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
            val imageFile = File(appDir, "DEBUG_${timeStamp}_$suffix.jpg")

            // Salva o bitmap como um arquivo JPG
            val fos = FileOutputStream(imageFile)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
            fos.flush()
            fos.close()
            Log.d("PlateRecognitionDebug", "Imagem de debug salva em: ${imageFile.absolutePath}")
        } catch (e: Exception) {
            Log.e("PlateRecognitionDebug", "Erro ao salvar imagem de debug: ${e.message}", e)
        }
    }


    private fun postprocessOutput(outputBuffer: FloatBuffer, correctedBitmap: Bitmap): String {
        // 1. Definimos um resultado padrão.
        // Ele será alterado se encontrarmos uma placa.
        var resultString = "Nenhuma placa detectada."

        outputBuffer.rewind()
        val rawOutput = Array(outputNumFeaturesPerBox) { FloatArray(outputNumBoxes) }
        for (featureIdx in 0 until outputNumFeaturesPerBox) {
            for (boxIdx in 0 until outputNumBoxes) {
                rawOutput[featureIdx][boxIdx] = outputBuffer.get()
            }
        }

        val detections = mutableListOf<Detection>()
        val confidenceThreshold = 0.6f // Mantendo o limiar mais alto
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

        val filteredByAspectRatio = finalDetections.filter {
            val aspectRatio = it.width / it.height
            aspectRatio > 1.5f
        }
        Log.d("PlateRecognition", "Numero de detecoes filtradas por aspect ratio: ${filteredByAspectRatio.size}")

        // 2. Se encontrarmos um candidato válido após todos os filtros, entramos neste bloco.
        if (filteredByAspectRatio.isNotEmpty()) {
            val bestDetection = filteredByAspectRatio.first()
            Log.d("PlateRecognition", "Placa final detectada com confianca ${bestDetection.confidence}")

            if (lastFrameBitmap != null) {
                // Coordenadas da caixa na imagem 640x640
                val boxOnYoloInput = RectF(
                    (bestDetection.xCenter - bestDetection.width / 2) * modelInputWidth,
                    (bestDetection.yCenter - bestDetection.height / 2) * modelInputHeight,
                    (bestDetection.xCenter + bestDetection.width / 2) * modelInputWidth,
                    (bestDetection.yCenter + bestDetection.height / 2) * modelInputHeight
                )
                val yoloViewBitmap = drawRectOnBitmap(lastFrameBitmap!!, boxOnYoloInput, android.graphics.Color.RED)
                saveBitmapForDebug(yoloViewBitmap, "00_YoloInput_With_Box.jpg")
            }

            val plateBox = convertYoloToPixelCoords(bestDetection, correctedBitmap)

            val cropAreaOnOriginal = RectF(plateBox.x1, plateBox.y1, plateBox.x2, plateBox.y2)
            val originalViewBitmap = drawRectOnBitmap(correctedBitmap, cropAreaOnOriginal, android.graphics.Color.GREEN)
            saveBitmapForDebug(originalViewBitmap, "01_Original_With_CropArea.jpg")

            try {
                val boxWidth = plateBox.x2 - plateBox.x1
                val boxHeight = plateBox.y2 - plateBox.y1
                val horizontalPadding = boxWidth * 0.10f
                val verticalPadding = boxHeight * 0.10f
                val refinedX1 = plateBox.x1 + horizontalPadding
                val refinedY1 = plateBox.y1 + verticalPadding
                val refinedX2 = plateBox.x2 - horizontalPadding
                val refinedY2 = plateBox.y2 - verticalPadding

                val x = refinedX1.toInt().coerceIn(0, correctedBitmap.width)
                val y = refinedY1.toInt().coerceIn(0, correctedBitmap.height)
                val width = (refinedX2 - refinedX1).toInt().coerceAtMost(correctedBitmap.width - x).coerceAtLeast(0)
                val height = (refinedY2 - refinedY1).toInt().coerceAtMost(correctedBitmap.height - y).coerceAtLeast(0)

                if (width >= 32 && height >= 32) {
                    val croppedBitmap = Bitmap.createBitmap(correctedBitmap, x, y, width, height)
                    saveBitmapForDebug(croppedBitmap, "01_cropped.jpg")

                    val ocrReadyBitmap = preprocessBitmapForOcr(croppedBitmap)
                    saveBitmapForDebug(ocrReadyBitmap, "02_ocr_readyGray.jpg")

                    val recognizedText = recognizeTextFromBitmap(ocrReadyBitmap)
                    val finalPlate = filterAndFormatPlateText(recognizedText)

                    Log.d("PlateRecognition", "Texto cru do OCR: $recognizedText")
                    Log.d("PlateRecognition", "Placa final formatada: $finalPlate")

                    // 3. Em vez de retornar, nós atualizamos nossa variável de resultado.
                    resultString = "$finalPlate"
                } else {
                    Log.e("PlateRecognition", "Erro de OCR: Imagem recortada muito pequena (w=$width, h=$height).")
                    resultString = "Erro de OCR: Imagem recortada muito pequena."
                }
            } catch (e: Exception) {
                Log.e("PlateRecognition", "Erro ao recortar imagem ou no OCR: ${e.message}", e)
                resultString = "Erro ao fazer OCR: ${e.message}"
            }
        }
        // Se a lista (filteredByAspectRatio) estiver vazia, o código simplesmente pula este bloco 'if'
        // e a 'resultString' mantém seu valor padrão "Nenhuma placa detectada."

        // 4. No final da função, temos um único ponto de saída que retorna o resultado.
        return resultString
    }

    private fun preprocessBitmapForOcr(bitmap: Bitmap): Bitmap {
        // 1. Converte para escala de cinza
        val grayBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(grayBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
        canvas.drawBitmap(bitmap, 0f, 0f, paint)

        // Salva a imagem em escala de cinza para comparação no debug
        saveBitmapForDebug(grayBitmap, "02_ocr_readyGray.jpg")

        // 2. Aplica a Binarização (Thresholding) para criar uma imagem preto e branco
        val binaryBitmap = Bitmap.createBitmap(grayBitmap.width, grayBitmap.height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(grayBitmap.width * grayBitmap.height)
        grayBitmap.getPixels(pixels, 0, grayBitmap.width, 0, 0, grayBitmap.width, grayBitmap.height)

        val threshold = 128 // Ponto de corte: pixels mais escuros viram preto, mais claros viram branco.
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val red = (pixel shr 16) and 0xFF // Em escala de cinza, R=G=B
            if (red < threshold) {
                pixels[i] = android.graphics.Color.BLACK
            } else {
                pixels[i] = android.graphics.Color.WHITE
            }
        }
        binaryBitmap.setPixels(pixels, 0, binaryBitmap.width, 0, 0, binaryBitmap.width, binaryBitmap.height)

        // Salva a imagem binarizada para você ver o resultado incrível no debug
        saveBitmapForDebug(binaryBitmap, "03_OcrInput_Binary.jpg")

        return binaryBitmap
    }
    

    private fun convertYoloToPixelCoords(detection: Detection, correctedBitmap: Bitmap): BoundingBox {
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
        val lines = rawText.uppercase().lines()
        val longestLine = lines
            .map { it.replace("[^A-Z0-9]".toRegex(), "") }
            .maxByOrNull { it.length } ?: ""

        // A validação de 7 caracteres é um excelente primeiro filtro.
        if (longestLine.length != 7) {
            return "Padrão não reconhecido: '$longestLine'"
        }

        // --- LÓGICA DE PRÉ-CORREÇÃO (mantida para robustez) ---
        var chars = longestLine.toCharArray()
        // As 3 primeiras posições DEVEM ser letras.
        for (i in 0..2) {
            if (chars[i] == '0') chars[i] = 'O'
            if (chars[i] == '1') chars[i] = 'I'
            if (chars[i] == '5') chars[i] = 'S'
        }
        val cleanText = String(chars)
        // --- FIM DA PRÉ-CORREÇÃO ---

        // Definimos as Regex para os dois padrões brasileiros
        val mercosulRegex = "[A-Z]{3}[0-9][A-Z][0-9]{2}".toRegex()
        val antigoRegex = "[A-Z]{3}[0-9]{4}".toRegex()

        // --- NOVA LÓGICA DE VALIDAÇÃO ---
        // Verificamos se o texto limpo CORRESPONDE INTEIRAMENTE a um dos padrões.

        // 1. Testa o Padrão Mercosul
        if (cleanText.matches(mercosulRegex)) {
            Log.d("PlateFormatter", "Padrão Mercosul detectado: $cleanText")
            return cleanText
        }

        // 2. Se não for Mercosul, testa o Padrão Antigo
        if (cleanText.matches(antigoRegex)) {
            Log.d("PlateFormatter", "Padrão Antigo detectado: $cleanText")
            return cleanText
        }

        // Se não corresponder a nenhum dos dois, o padrão é inválido.
        return "Padrão não reconhecido: '$cleanText'"
    }

    private fun drawRectOnBitmap(bitmap: Bitmap, rect: RectF, color: Int): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = android.graphics.Canvas(mutableBitmap)
        val paint = Paint()
        paint.color = color
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4.0f
        canvas.drawRect(rect, paint)
        return mutableBitmap
    }

    private fun fixOrientation(bitmap: Bitmap, imagePath: String): Bitmap {
        val ei = ExifInterface(imagePath)
        val orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED)

        // Com base na orientação dos metadados, criamos uma matriz de transformação
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            // Adicione outros casos se necessário, mas estes são os principais
            else -> return bitmap // Nenhuma rotação necessária, retorna o bitmap original
        }

        // Cria e retorna um novo bitmap rotacionado
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        // Configura a matriz para aplicar a rotação
        matrix.postRotate(angle)
        // Cria e retorna um novo bitmap a partir do original, aplicando a matriz de rotação
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }


    override fun onDestroy() {
        super.onDestroy()
        tfliteInterpreter?.close()
    }

    private fun isBlurry(bitmap: Bitmap, threshold: Double = 100.0): Boolean{
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat) // Converte Bitmap para o formato do OpenCV

        val grayMat = Mat()
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY) // Converte para escala de cinza

        val laplacianMat = Mat()
        Imgproc.Laplacian(grayMat, laplacianMat, org.opencv.core.CvType.CV_64F) // Aplica o filtro Laplaciano

        val meanStdDev = MatOfDouble()
        org.opencv.core.Core.meanStdDev(laplacianMat, MatOfDouble(), meanStdDev) // Calcula a variância (desvio padrão ao quadrado)

        val variance = Math.pow(meanStdDev.get(0, 0)[0], 2.0)
        Log.d("BlurDetection", "Variância da imagem: $variance")

        return variance < threshold // Se a variância for menor que o limiar, a imagem está borrada
    }
}