import 'package:flutter/material.dart';
import 'package:camera/camera.dart'; // Import para a câmera
import 'package:path_provider/path_provider.dart'; // Para diretório temporário
import 'package:path/path.dart' show join; // Para juntar caminhos
import 'dart:io'; // Para operações de arquivo
import 'package:flutter/services.dart'; // Adicionado: Import necessário para MethodChannel
import 'package:permission_handler/permission_handler.dart'; // Para solicitar permissões

// Esta lista será preenchida com as câmeras disponíveis no dispositivo
List<CameraDescription> cameras = [];

// ====================================================================

const platform = MethodChannel('com.example.ocr_recog/plate_recognition');

Future<String> recognizePlate(String imagePath) async {
  try {
    // Chama o método 'recognizePlate' no lado nativo, passando o caminho da imagem como argumento.
    final String result = await platform.invokeMethod('recognizePlate', {'imagePath': imagePath});
    return result;
  } on PlatformException catch (e) {
    // Captura erros que podem vir do lado nativo
    print("Erro nativo ao reconhecer placa: '${e.code}': '${e.message}'.");
    return "Erro: ${e.message}"; // Retorna uma mensagem de erro para ser exibida na UI
  } catch (e) {
    // Captura outros erros inesperados
    print("Erro inesperado ao chamar reconhecimento nativo: $e");
    return "Erro inesperado: $e";
  }
}
// ====================================================================

Future<void> main() async {
  // Garante que os widgets do Flutter estejam inicializados
  WidgetsFlutterBinding.ensureInitialized();

  // Obtém a lista de câmeras disponíveis no dispositivo
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    print('Erro ao obter câmeras: ${e.code}, ${e.description}');
  }

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Reconhecimento de Placas',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      home: const PlateRecognitionScreen(),
    );
  }
}

Future<String?> recognizePlateFromStream(CameraImage image) async {
  try {
    // O formato YUV420 tem 3 "planos" de bytes. Vamos juntá-los.
    final WriteBuffer allBytes = WriteBuffer();
    for (final Plane plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();

    // Enviamos os bytes e as dimensões da imagem para o Kotlin
    final Map<String, dynamic> arguments = {
      'bytes': bytes,
      'height': image.height,
      'width': image.width,
    };
    
    // Usamos um novo nome de método para o stream
    final String? result = await platform.invokeMethod('recognizePlateFromStream', arguments);
    return result;
  } catch (e) {
    print("Erro no stream: $e");
    return null;
  }
}

class PlateRecognitionScreen extends StatefulWidget {
  const PlateRecognitionScreen({super.key});

  @override
  State<PlateRecognitionScreen> createState() => _PlateRecognitionScreenState();
}

class _PlateRecognitionScreenState extends State<PlateRecognitionScreen> {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  String _recognizedPlate = "Aponte para uma placa..."; // Texto inicial mais apropriado
  bool _isProcessing = false; // Flag para controlar o processamento do stream

  @override
  void initState() {
    super.initState();
    // Garante que temos câmeras disponíveis
    if (cameras.isNotEmpty) {
      _controller = CameraController(
        cameras[0], // Usa a primeira câmera (geralmente a traseira)
        ResolutionPreset.medium, // 'medium' é ideal para performance de stream
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420, // Formato essencial para o stream
      );

      _initializeControllerFuture = _controller!.initialize().then((_) {
        // Este 'if' é uma boa prática para evitar erros se o usuário sair
        // da tela antes da câmera inicializar.
        if (!mounted) {
          return;
        }
        // Assim que a câmera estiver pronta, iniciamos o fluxo de imagens.
        _startImageStream();
        setState(() {}); // Para reconstruir a UI com o preview da câmera
      });
    } else {
      // Lida com o caso de não haver câmeras
      _recognizedPlate = "Nenhuma câmera disponível.";
      _initializeControllerFuture = Future.value();
    }
  }

  // A nova função que gerencia o fluxo de imagens da câmera
  void _startImageStream() {
    _controller!.startImageStream((CameraImage image) {
      // Se um frame já está sendo processado pelo lado nativo, pulamos este.
      // Isso evita sobrecarregar o processador.
      if (_isProcessing) return;

      // Seta a flag para 'true' para bloquear os próximos frames.
      _isProcessing = true;

      // Chama a função de comunicação nativa para processar o frame
      recognizePlateFromStream(image).then((result) {
        // Antes de atualizar a UI, garante que o widget ainda está "vivo"
        if (mounted && result != null && result.isNotEmpty) {
          setState(() {
            _recognizedPlate = result;
          });
        }
      }).whenComplete(() {
        // IMPORTANTE: Sempre seta a flag de volta para 'false' ao final,
        // independentemente se houve sucesso ou falha.
        _isProcessing = false;
      });
    });
  }

  @override
  void dispose() {
    // É CRUCIAL parar o stream de imagens antes de descartar o controlador
    // para liberar recursos e economizar bateria.
    _controller?.stopImageStream();
    _controller?.dispose();
    super.dispose();
  }

  // A função _takePicture() foi removida pois não é mais necessária.

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Reconhecimento de Placas'),
      ),
      body: Column(
        children: [
          // --- Área da Câmera (70% da tela) ---
          Expanded(
            flex: 7,
            child: FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  // Garante que o controlador não é nulo e está inicializado
                  if (_controller != null && _controller!.value.isInitialized) {
                    return CameraPreview(_controller!);
                  } else {
                    return const Center(child: Text("Não foi possível iniciar a câmera."));
                  }
                } else {
                  // Mostra um indicador de carregamento enquanto a câmera inicializa
                  return const Center(child: CircularProgressIndicator());
                }
              },
            ),
          ),
          // --- Área de Texto (30% da tela) ---
          Expanded(
            flex: 3,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  TextField(
                    // Usamos um TextEditingController para que o texto seja atualizado
                    // eficientemente pela chamada do setState.
                    controller: TextEditingController(text: _recognizedPlate),
                    decoration: const InputDecoration(
                      labelText: 'Placa Reconhecida',
                      border: OutlineInputBorder(),
                    ),
                    readOnly: true,
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, letterSpacing: 2),
                  ),
                  const SizedBox(height: 16),
                  // O botão foi removido e substituído por uma instrução.
                  const Text(
                    "O reconhecimento é automático.",
                    style: TextStyle(color: Colors.grey),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}