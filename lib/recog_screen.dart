import 'package:flutter/material.dart';
import 'package:camera/camera.dart'; // Import para a câmera
import 'package:path_provider/path_provider.dart'; // Para diretório temporário
import 'package:path/path.dart' show join; // Para juntar caminhos
import 'dart:io'; // Para operações de arquivo
import 'package:flutter/services.dart'; // Adicionado: Import necessário para MethodChannel

// Esta lista será preenchida com as câmeras disponíveis no dispositivo
List<CameraDescription> cameras = [];

// ====================================================================
// Adicionado: Definição do MethodChannel e função para chamar o nativo
// O nome do canal DEVE ser EXATAMENTE o mesmo que você usará no Kotlin!
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

class PlateRecognitionScreen extends StatefulWidget {
  const PlateRecognitionScreen({super.key});

  @override
  State<PlateRecognitionScreen> createState() => _PlateRecognitionScreenState();
}

class _PlateRecognitionScreenState extends State<PlateRecognitionScreen> {
  CameraController? _controller; // Controlador da câmera
  Future<void>? _initializeControllerFuture; // Futuro para a inicialização da câmera
  String _recognizedPlate = "Placa aparecerá aqui..."; // Campo para exibir a placa

  @override
  void initState() {
    super.initState();
    // Inicializa a câmera
    if (cameras.isNotEmpty) {
      _controller = CameraController(
        // Use a primeira câmera disponível (geralmente a traseira)
        cameras[0],
        // Defina a resolução da câmera. Medium é um bom balanço para reconhecimento.
        ResolutionPreset.medium,
        enableAudio: false, // Não precisamos de áudio para este app
      );

      // Inicializa o futuro do controlador
      _initializeControllerFuture = _controller!.initialize();
    } else {
      _recognizedPlate = "Nenhuma câmera disponível.";
      _initializeControllerFuture = Future.value(); // Resolve imediatamente
    }
  }

  @override
  void dispose() {
    // Descarte o controlador quando o widget for removido da árvore
    _controller?.dispose();
    super.dispose();
  }

  // Função para tirar foto (e agora chamará o reconhecimento)
  Future<void> _takePicture() async {
    try {
      // Garante que a câmera foi inicializada antes de tirar a foto
      await _initializeControllerFuture;

      // Constrói o caminho para onde a imagem capturada será salva
      final path = join(
        (await getTemporaryDirectory()).path,
        '${DateTime.now()}.png',
      );

      // Tira a foto e a salva no caminho especificado
      XFile image = await _controller!.takePicture();
      await image.saveTo(path); // Salva a imagem no diretório temporário

      setState(() {
        _recognizedPlate = "Capturando e processando..."; // Feedback imediato
      });

      // ====================================================================
      // Modificado: Chamar a função nativa de reconhecimento
      final String recognizedResult = await recognizePlate(path);
      // ====================================================================

      setState(() {
        _recognizedPlate = recognizedResult; // Atualiza o campo de texto com o resultado do reconhecimento
      });
    } catch (e) {
      print("Erro ao tirar foto: $e");
      setState(() {
        _recognizedPlate = "Erro ao tirar foto: ${e.toString()}";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Reconhecimento de Placas'),
      ),
      body: Column(
        children: [
          // --- Área da Câmera (60-70% da tela) ---
          Expanded(
            flex: 7, // Ocupa 70% do espaço disponível
            child: FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  // Se o Future estiver completo, mostra a pré-visualização da câmera
                  return CameraPreview(_controller!);
                } else {
                  // Caso contrário, mostra um indicador de carregamento
                  return const Center(child: CircularProgressIndicator());
                }
              },
            ),
          ),
          // --- Área de Texto e Botão de Captura (30-40% da tela) ---
          Expanded(
            flex: 3, // Ocupa 30% do espaço restante
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  TextField(
                    controller: TextEditingController(text: _recognizedPlate),
                    decoration: const InputDecoration(
                      labelText: 'Placa Reconhecida',
                      border: OutlineInputBorder(),
                    ),
                    readOnly: true, // Campo somente leitura
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: _takePicture, // Chama a função de tirar foto
                    child: const Text('Capturar Placa'),
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