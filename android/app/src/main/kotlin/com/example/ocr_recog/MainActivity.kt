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

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.ocr_recog/plate_recognition"
    private var tfliteInterpreter: Interpreter? = null
    private val MODEL_PATH = "license-plate-finetune-v1s_int8.tflite"

    // --- DETALHES DE ENTRADA DO MODELO ---
    private val modelInputWidth: Int = 640
    private val modelInputHeight: Int = 640
    private val modelInputChannels: Int = 3
    private val modelInputDataType: DataType = DataType.FLOAT32
    // A normalização para [0, 1] será feita manualmente (pixel / 255.0f)
    private val modelInputNormalizeMean: Float = 0.0f
    private val modelInputNormalizeStd: Float = 1.0f

    // --- DETALHES DE SAÍDA DO MODELO ---
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
                    try {
                        if (tfliteInterpreter == null) {
                            loadModelInterpreter()
                        }
                        val bitmap = BitmapFactory.decodeFile(imagePath)
                        if (bitmap == null) {
                            result.error("IMAGE_LOAD_FAILED", "Não foi possível carregar a imagem do caminho fornecido.", null)
                            return@setMethodCallHandler
                        }
                        val inputBuffer = preprocessBitmap(bitmap)
                        val outputFloatBuffer = ByteBuffer.allocateDirect(modelOutputBufferLength).order(ByteOrder.nativeOrder()).asFloatBuffer()
                        val outputMap = mutableMapOf<Int, Any>()
                        outputMap[0] = outputFloatBuffer
                        tfliteInterpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
                        val plateText = postprocessOutput(outputFloatBuffer)
                        result.success(plateText)
                    } catch (e: Exception) {
                        Log.e("PlateRecognition", "Erro durante o reconhecimento da placa: ${e.message}", e)
                        result.error("RECOGNITION_ERROR", "Erro no processamento da placa: ${e.message}", e.toString())
                    }
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
            val assetFileDescriptor = assets.openFd("license-plate-finetune-v1s_int8.tflite")
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

    // --- FUNÇÃO preprocessBitmap MANUAL (versão 2.0 mais robusta) ---
    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        // Redimensiona o bitmap
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, modelInputWidth, modelInputHeight, true)

        // Cria um ByteBuffer temporário para os pixels
        val pixelBuffer = ByteBuffer.allocateDirect(modelInputWidth * modelInputHeight * 4) // 4 bytes por pixel (ARGB_8888)
        scaledBitmap.copyPixelsToBuffer(pixelBuffer)
        pixelBuffer.rewind()

        // Cria o ByteBuffer final no formato Float32 RGB
        val inputBuffer = ByteBuffer.allocateDirect(modelInputWidth * modelInputHeight * modelInputChannels * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        for (i in 0 until modelInputWidth * modelInputHeight) {
            val pixelValue = pixelBuffer.getInt() // Lê o pixel como um inteiro ARGB_8888

            // Extrai R, G, B
            val r = (pixelValue shr 16 and 0xFF).toFloat()
            val g = (pixelValue shr 8 and 0xFF).toFloat()
            val b = (pixelValue and 0xFF).toFloat()

            // Normaliza para [0, 1] e coloca no buffer
            inputBuffer.putFloat(r / 255.0f)
            inputBuffer.putFloat(g / 255.0f)
            inputBuffer.putFloat(b / 255.0f)
        }
        inputBuffer.rewind()
        return inputBuffer
    }

    private fun postprocessOutput(outputBuffer: FloatBuffer): String {
        outputBuffer.rewind()

        val rawOutput = Array(outputNumFeaturesPerBox) { FloatArray(outputNumBoxes) }

        for (featureIdx in 0 until outputNumFeaturesPerBox) {
            for (boxIdx in 0 until outputNumBoxes) {
                rawOutput[featureIdx][boxIdx] = outputBuffer.get()
            }
        }

        var numHighConfidenceDetections = 0
        val confidenceThreshold = 0.25f
        for (i in 0 until outputNumBoxes) {
            val confidence = rawOutput[4][i]
            if (confidence > confidenceThreshold) {
                numHighConfidenceDetections++
            }
        }

        return "Detectadas $numHighConfidenceDetections objetos (Pós-processamento YOLO pendente)."
    }

    override fun onDestroy() {
        super.onDestroy()
        tfliteInterpreter?.close()
    }
}