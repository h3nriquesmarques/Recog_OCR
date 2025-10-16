import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

// A lista de câmeras é a única variável que precisa ser global,
// pois é inicializada no main() antes do app rodar.
late List<CameraDescription> cameras;

// O canal de comunicação também pode ficar aqui.
const platform = MethodChannel('com.example.ocr_recog/plate_recognition');

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
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

// ====================================================================
// TODA A LÓGICA AGORA ESTÁ DENTRO DESTA CLASSE, CORRIGIDA
// ====================================================================

class _PlateRecognitionScreenState extends State<PlateRecognitionScreen> {
  // --- VARIÁVEIS DE ESTADO ---
  // Todas as variáveis que controlam a UI e a lógica ficam aqui dentro.
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  bool _isProcessing = false;
  Timer? _validationTimer;

  String _lastRecognizedPlate = "";
  bool _isSearching = true;
  final List<String> _savedPlates = [];
  // --------------------------------------------------------------------

  @override
  void initState() {
    super.initState();
    if (cameras.isNotEmpty) {
      _controller = CameraController(
        cameras[0],
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      _initializeControllerFuture = _controller!.initialize().then((_) {
        if (!mounted) return;
        _startImageStream();
        setState(() {});
      });
    }
  }

  @override
  void dispose() {
    _validationTimer?.cancel();
    if (_controller != null && _controller!.value.isStreamingImages) {
      _controller?.stopImageStream();
    }
    _controller?.dispose();
    super.dispose();
  }

  // --- FUNÇÕES DE LÓGICA E CONTROLE (COM A CORREÇÃO PRINCIPAL) ---

  void _startImageStream() {
    if (_controller != null && !_controller!.value.isStreamingImages) {
      _controller!.startImageStream((CameraImage image) {
        if (_isProcessing || !_isSearching) return;
        _isProcessing = true;

        _recognizePlateFromStream(image).then((result) {
          if (!mounted || result == null || result.isEmpty || !_isSearching) return;

          // --- ESTA É A CORREÇÃO PRINCIPAL ---
          // Verificamos se o resultado é um sucesso ANTES de parar o stream.
          final bool isSuccess = !result.contains('Erro') && !result.contains('Padrão') && !result.contains('Nenhuma placa detectada');

          if (isSuccess) {
            // SUCESSO: Paramos o stream e iniciamos a validação de 10s.
            _controller?.stopImageStream();
            setState(() {
              _isSearching = false;
              _lastRecognizedPlate = result;
            });

            _validationTimer = Timer(const Duration(seconds: 10), () {
              print("Tempo de validação esgotado! Retomando a busca...");
              _retrySearch();
            });
          }
          // Se 'isSuccess' for falso (um erro de reconhecimento), nada acontece aqui.
          // O stream NÃO é parado e a busca continua no próximo frame,
          // exatamente como você pediu.
          // ------------------------------------

        }).whenComplete(() {
          _isProcessing = false;
        });
      });
    }
  }

  void _confirmPlate() {
    _validationTimer?.cancel();
    setState(() {
      final bool isSuccess = !_lastRecognizedPlate.contains('Erro') && !_lastRecognizedPlate.contains('Padrão');
      if (isSuccess && _lastRecognizedPlate.isNotEmpty && !_savedPlates.contains(_lastRecognizedPlate)) {
        _savedPlates.add(_lastRecognizedPlate);
      }
      _isSearching = true;
      _lastRecognizedPlate = "";
    });
    _startImageStream();
  }

  void _retrySearch() {
    _validationTimer?.cancel();
    if (mounted) {
      setState(() {
        _isSearching = true;
        _lastRecognizedPlate = "";
      });
    }
    _startImageStream();
  }

  // --- FUNÇÃO DE COMUNICAÇÃO NATIVA ---

  Future<String?> _recognizePlateFromStream(CameraImage image) async {
    try {
      final WriteBuffer allBytes = WriteBuffer();
      for (final Plane plane in image.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();
      
      final int rotation = _controller!.value.deviceOrientation.index;

      final Map<String, dynamic> arguments = {
        'bytes': bytes,
        'height': image.height,
        'width': image.width,
        'rotation': rotation,
      };

      final String? result = await platform.invokeMethod('recognizePlateFromStream', arguments);
      return result;
    } catch (e) {
      print("Erro no stream: $e");
      return null;
    }
  }

  // --- WIDGETS DA UI ---

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Reconhecimento de Placas'),
      ),
      body: Column(
        children: [
          Expanded(
            flex: 7,
            child: FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  if (_controller != null && _controller!.value.isInitialized) {
                    return Stack(
                      alignment: Alignment.bottomCenter,
                      children: [
                        CameraPreview(_controller!),
                        _buildResultsOverlay(),
                      ],
                    );
                  } else {
                    return const Center(child: Text("Câmera indisponível."));
                  }
                } else {
                  return const Center(child: CircularProgressIndicator());
                }
              },
            ),
          ),
          Expanded(
            flex: 3,
            child: Container(
              color: Colors.grey[200],
              child: ListView.builder(
                itemCount: _savedPlates.length,
                itemBuilder: (context, index) {
                  final plate = _savedPlates[index];
                  return ListTile(
                    leading: const Icon(Icons.directions_car),
                    title: Text(plate, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    trailing: IconButton(
                      icon: const Icon(Icons.copy),
                      onPressed: () {
                        Clipboard.setData(ClipboardData(text: plate));
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('Placa copiada!')),
                        );
                      },
                    ),
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultsOverlay() {
    if (_isSearching) {
      return Positioned(
        bottom: 0,
        left: 0,
        right: 0,
        child: Container(
          padding: const EdgeInsets.all(12.0),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.5),
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(16),
              topRight: Radius.circular(16),
            ),
          ),
          child: const Text(
            'Buscando placa...',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white, fontSize: 18),
          ),
        ),
      );
    }

    final bool isSuccess = !_lastRecognizedPlate.contains('Erro') && !_lastRecognizedPlate.contains('Padrão');

    if (isSuccess) {
      return Positioned(
        bottom: 0,
        left: 0,
        right: 0,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(16),
              topRight: Radius.circular(16),
            ),
            boxShadow: [
              BoxShadow(color: Colors.black.withOpacity(0.2), spreadRadius: 2, blurRadius: 5),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconButton(
                icon: const Icon(Icons.close, color: Colors.red, size: 30),
                onPressed: _retrySearch,
                tooltip: 'Tentar Novamente',
              ),
              Expanded(
                child: Text(
                  _lastRecognizedPlate,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, letterSpacing: 2),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              IconButton(
                icon: const Icon(Icons.check, color: Colors.green, size: 30),
                onPressed: _confirmPlate,
                tooltip: 'Confirmar',
              ),
            ],
          ),
        ),
      );
    } else {
      return Positioned(
        bottom: 0,
        left: 0,
        right: 0,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
          decoration: BoxDecoration(
            color: Colors.red[700],
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(16),
              topRight: Radius.circular(16),
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconButton(
                icon: const Icon(Icons.refresh, color: Colors.white, size: 30),
                onPressed: _retrySearch,
                tooltip: 'Tentar Novamente',
              ),
              Expanded(
                child: Text(
                  _lastRecognizedPlate,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white, fontSize: 16),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 48),
            ],
          ),
        ),
      );
    }
  }
}