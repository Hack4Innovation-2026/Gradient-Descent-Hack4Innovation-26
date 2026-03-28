import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'dart:math';

const List<String> kClasses = [
  "ALL_MOTOR_VEHICLE_PROHIBITED", "AXLE_LOAD_LIMIT", "BARRIER_AHEAD",
  "BULLOCK_AND_HANDCART_PROHIBITED", "BULLOCK_PROHIBITED", "CATTLE",
  "COMPULSARY_AHEAD", "COMPULSARY_AHEAD_OR_TURN_LEFT",
  "COMPULSARY_AHEAD_OR_TURN_RIGHT", "COMPULSARY_CYCLE_TRACK",
  "COMPULSARY_KEEP_LEFT", "COMPULSARY_KEEP_RIGHT",
  "COMPULSARY_MINIMUM_SPEED", "COMPULSARY_SOUND_HORN",
  "COMPULSARY_TURN_LEFT", "COMPULSARY_TURN_LEFT_AHEAD",
  "COMPULSARY_TURN_RIGHT", "COMPULSARY_TURN_RIGHT_AHEAD",
  "CROSS_ROAD", "CYCLE_CROSSING", "CYCLE_PROHIBITED", "DANGEROUS_DIP",
  "DIRECTION", "FALLING_ROCKS", "FERRY", "GAP_IN_MEDIAN", "GIVE_WAY",
  "GUARDED_LEVEL_CROSSING", "HANDCART_PROHIBITED", "HEIGHT_LIMIT",
  "HORN_PROHIBITED", "HUMP_OR_ROUGH_ROAD", "LEFT_HAIR_PIN_BEND",
  "LEFT_HAND_CURVE", "LEFT_REVERSE_BEND", "LEFT_TURN_PROHIBITED",
  "LENGTH_LIMIT", "LOAD_LIMIT", "LOOSE_GRAVEL", "MEN_AT_WORK",
  "NARROW_BRIDGE", "NARROW_ROAD_AHEAD", "NO_ENTRY", "NO_PARKING",
  "NO_STOPPING_OR_STANDING", "OVERTAKING_PROHIBITED", "PASS_EITHER_SIDE",
  "PEDESTRIAN_CROSSING", "PEDESTRIAN_PROHIBITED",
  "PRIORITY_FOR_ONCOMING_VEHICLES", "QUAY_SIDE_OR_RIVER_BANK",
  "RESTRICTION_ENDS", "RIGHT_HAIR_PIN_BEND", "RIGHT_HAND_CURVE",
  "RIGHT_REVERSE_BEND", "RIGHT_TURN_PROHIBITED", "ROAD_WIDENS_AHEAD",
  "ROUNDABOUT", "SCHOOL_AHEAD", "SIDE_ROAD_LEFT", "SIDE_ROAD_RIGHT",
  "SLIPPERY_ROAD", "SPEED_LIMIT_15", "SPEED_LIMIT_20", "SPEED_LIMIT_30",
  "SPEED_LIMIT_40", "SPEED_LIMIT_5", "SPEED_LIMIT_50", "SPEED_LIMIT_60",
  "SPEED_LIMIT_70", "SPEED_LIMIT_80", "STAGGERED_INTERSECTION",
  "STEEP_ASCENT", "STEEP_DESCENT", "STOP", "STRAIGHT_PROHIBITED",
  "TONGA_PROHIBITED", "TRAFFIC_SIGNAL", "TRUCK_PROHIBITED", "TURN_RIGHT",
  "T_INTERSECTION", "UNGUARDED_LEVEL_CROSSING", "U_TURN_PROHIBITED",
  "WIDTH_LIMIT", "Y_INTERSECTION"
];

const int kClassifierInputSize = 64;
const int kDetectorInputSize   = 640;
const double kConfThreshold    = 0.05;
const double kClassThreshold   = 0.30;

// ─────────────────────────────────────────────
// BoundingBox  —  coords are normalised 0-1 straight from model
// ─────────────────────────────────────────────
// ─────────────────────────────────────────────
// BoundingBox  —  coords are in PIXEL space (0–640)
// ─────────────────────────────────────────────
class BoundingBox {
  final double cx, cy, w, h, confidence;
  BoundingBox(this.cx, this.cy, this.w, this.h, this.confidence);

  /// cx,cy,w,h are in detector input pixel space (0–640)
  /// Scale them to original image dimensions
  Rect toRect(int imgW, int imgH) {
  final left   = (cx * imgW) - (w * imgW / 2);
  final top    = (cy * imgH) - (h * imgH / 2);
  final right  = (cx * imgW) + (w * imgW / 2);
  final bottom = (cy * imgH) + (h * imgH / 2);

  return Rect.fromLTRB(
    left.clamp(0.0, imgW.toDouble()),
    top.clamp(0.0, imgH.toDouble()),
    right.clamp(0.0, imgW.toDouble()),
    bottom.clamp(0.0, imgH.toDouble()),
  );
}
}
// ─────────────────────────────────────────────
// Detector
// ─────────────────────────────────────────────
class SignboardDetector {
  Interpreter? _interpreter;

  Future<void> load() async {
    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(
      'assets/last_float33.tflite',
      options: options,
    );
    _interpreter!.allocateTensors();
  }

  BoundingBox? detect(img.Image source) {
    if (_interpreter == null) return null;

    // ── Resize to 640×640 ──
    final resized = img.copyResize(
      source,
      width:  kDetectorInputSize,
      height: kDetectorInputSize,
      interpolation: img.Interpolation.linear,
    );

    // ── Build [1, 640, 640, 3] float input ──
    final input = List.generate(
      1, (_) => List.generate(
        kDetectorInputSize, (y) => List.generate(
          kDetectorInputSize, (x) {
            final p = resized.getPixel(x, y);
            return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
          },
        ),
      ),
    );

    // ── Output shape: [1, 5, 8400] ──
    const int NUM_BOXES = 8400;

final output = List.generate(
  1, (_) => List.generate(5, (_) => List.filled(NUM_BOXES, 0.0)),
);

    _interpreter!.run(input, output);

    final cxRow   = output[0][0];
    final cyRow   = output[0][1];
    final wRow    = output[0][2];
    final hRow    = output[0][3];
    final confRow = output[0][4]; // raw logits → need sigmoid
print("RAW CONF SAMPLE: ${confRow[0]}");
    double sigmoid(double x) => 1.0 / (1.0 + exp(-x));

    // ── DEBUG: inspect raw values once ──
    final topConfs = (List<double>.from(confRow)..sort((a, b) => b.compareTo(a)))
        .take(3).toList();
    print('[Detector] top-3 raw logits: $topConfs');
    print('[Detector] sample cx[0]=${cxRow[0].toStringAsFixed(4)}  '
          'cy[0]=${cyRow[0].toStringAsFixed(4)}  '
          'w[0]=${wRow[0].toStringAsFixed(4)}');

    BoundingBox? best;
    double bestConf = kConfThreshold;

    for (int i = 0; i < NUM_BOXES; i++) {
      final conf = sigmoid(confRow[i]); // ✅ sigmoid applied
      if (conf > bestConf) {
        // if (wRow[i] < 0.05 || hRow[i] < 0.05) continue;
  print("MAX CONF: ${confRow.reduce(max)}");
print("MIN CONF: ${confRow.reduce(min)}");
  bestConf = conf;
  best = BoundingBox(cxRow[i], cyRow[i], wRow[i], hRow[i], conf);
      }
    }

    if (best != null) {
      print('[Detector] best → cx=${best.cx.toStringAsFixed(3)} '
            'cy=${best.cy.toStringAsFixed(3)} '
            'w=${best.w.toStringAsFixed(3)} '
            'h=${best.h.toStringAsFixed(3)} '
            'conf=${best.confidence.toStringAsFixed(3)}');
    } else {
      print('[Detector] no box above threshold');
    }

    return best;
  }

  void dispose() => _interpreter?.close();
}

// ─────────────────────────────────────────────
// Classifier
// ─────────────────────────────────────────────
class TrafficSignClassifier {
  Interpreter? _interpreter;

  Future<void> load() async {
    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(
      'assets/lmodel.tflite',
      options: options,
    );
    _interpreter!.allocateTensors();
  }

  String? classify(img.Image source) {
    if (_interpreter == null) return null;

    final resized = img.copyResize(
      source,
      width:  kClassifierInputSize,
      height: kClassifierInputSize,
      interpolation: img.Interpolation.linear,
    );

    final input = List.generate(
      1, (_) => List.generate(
        kClassifierInputSize, (y) => List.generate(
          kClassifierInputSize, (x) {
            final p = resized.getPixel(x, y);
            return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
          },
        ),
      ),
    );

    final output = List.generate(1, (_) => List.filled(kClasses.length, 0.0));
    _interpreter!.run(input, output);

    final scores = output[0];
    int bestIdx = 0;
    double bestVal = scores[0];
    for (int i = 1; i < scores.length; i++) {
      if (scores[i] > bestVal) { bestVal = scores[i]; bestIdx = i; }
    }

    print('[Classifier] → ${kClasses[bestIdx]} ($bestVal)');

    if (bestVal < kClassThreshold) return 'Could not classify';
    return '${kClasses[bestIdx]} (${(bestVal * 100).toStringAsFixed(1)}%)';
  }

  void dispose() => _interpreter?.close();
}

// ─────────────────────────────────────────────
// Pipeline result
// ─────────────────────────────────────────────
class PipelineResult {
  final String label;
  final Rect?  box;
  final double detectorConf;
  PipelineResult(this.label, this.box, this.detectorConf);
}

// ─────────────────────────────────────────────
// Pipeline  —  detect → crop → classify
// ─────────────────────────────────────────────
class SignPipeline {
  final SignboardDetector    detector   = SignboardDetector();
  final TrafficSignClassifier classifier = TrafficSignClassifier();

  Future<void> load() async {
    await detector.load();
    await classifier.load();
  }

  PipelineResult run(img.Image frame) {
    // ── 1. Detect ──
    final box = detector.detect(frame);
    if (box == null) return PipelineResult('No sign detected', null, 0.0);

    // ── 2. Convert normalised box → pixel rect on ORIGINAL frame ──
    final rect = box.toRect(frame.width, frame.height);

    // Guard: skip if box is degenerate
    if (rect.width < 8 || rect.height < 8) {
      return PipelineResult('Detection box too small', null, box.confidence);
    }

    // ── 3. Crop with 10 % padding ──
    const pad = 0.10;
    final left   = (rect.left   - rect.width  * pad).clamp(0.0, frame.width.toDouble());
    final top    = (rect.top    - rect.height * pad).clamp(0.0, frame.height.toDouble());
    final right  = (rect.right  + rect.width  * pad).clamp(0.0, frame.width.toDouble());
    final bottom = (rect.bottom + rect.height * pad).clamp(0.0, frame.height.toDouble());

    final cropW = (right  - left).toInt().clamp(1, frame.width);
    final cropH = (bottom - top ).toInt().clamp(1, frame.height);

    print('[Pipeline] crop x=${left.toInt()} y=${top.toInt()} '
          'w=$cropW h=$cropH  (frame ${frame.width}×${frame.height})');

    final cropped = img.copyCrop(
      frame,
      x: left.toInt(),
      y: top.toInt(),
      width:  cropW,
      height: cropH,
    );

    // ── 4. Classify ──
    final label = classifier.classify(cropped) ?? 'Could not classify';
    return PipelineResult(label, rect, box.confidence);
  }

  void dispose() {
    detector.dispose();
    classifier.dispose();
  }
}

// ─────────────────────────────────────────────
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Traffic Sign Recognizer',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
          useMaterial3: true,
        ),
        home: HomeScreen(cameras: cameras),
      );
}

// ─────────────────────────────────────────────
// Home screen
// ─────────────────────────────────────────────
class HomeScreen extends StatelessWidget {
  final List<CameraDescription> cameras;
  const HomeScreen({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      appBar: AppBar(
        backgroundColor: Colors.deepPurple,
        title: const Text('Traffic Sign Recognizer',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          const Icon(Icons.traffic, size: 80, color: Colors.deepPurpleAccent),
          const SizedBox(height: 32),
          _ModeButton(
            icon: Icons.videocam,
            label: 'Live Camera',
            onTap: () => Navigator.push(context, MaterialPageRoute(
                builder: (_) => LiveCameraScreen(cameras: cameras))),
          ),
          const SizedBox(height: 16),
          _ModeButton(
            icon: Icons.camera_alt,
            label: 'Take Photo',
            onTap: () => Navigator.push(context, MaterialPageRoute(
                builder: (_) => const ImageClassifyScreen(
                    source: ImageSource.camera))),
          ),
          const SizedBox(height: 16),
          _ModeButton(
            icon: Icons.photo_library,
            label: 'Pick from Gallery',
            onTap: () => Navigator.push(context, MaterialPageRoute(
                builder: (_) => const ImageClassifyScreen(
                    source: ImageSource.gallery))),
          ),
        ]),
      ),
    );
  }
}

class _ModeButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  const _ModeButton(
      {required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) => SizedBox(
        width: 260,
        child: ElevatedButton.icon(
          onPressed: onTap,
          icon: Icon(icon),
          label: Text(label, style: const TextStyle(fontSize: 16)),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.deepPurple,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14)),
          ),
        ),
      );
}

// ─────────────────────────────────────────────
// Screen 1 — Pick / capture a single image
// ─────────────────────────────────────────────
class ImageClassifyScreen extends StatefulWidget {
  final ImageSource source;
  const ImageClassifyScreen({super.key, required this.source});

  @override
  State<ImageClassifyScreen> createState() => _ImageClassifyScreenState();
}

class _ImageClassifyScreenState extends State<ImageClassifyScreen> {
  final _pipeline = SignPipeline();

  Uint8List?     _imageBytes;
  PipelineResult? _result;
  bool           _loading = false;
  int?           _imgW;
  int?           _imgH;

  @override
  void initState() {
    super.initState();
    _pipeline.load().then((_) => _pickAndRun());
  }

  Future<void> _pickAndRun() async {
    final picked = await ImagePicker().pickImage(
      source: widget.source,
      imageQuality: 90,
      maxWidth: 1280,
      maxHeight: 1280,
    );
    if (picked == null) return;

    setState(() { _loading = true; _result = null; });

    try {
      final bytes   = await File(picked.path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) { setState(() => _loading = false); return; }

      _imgW = decoded.width;
      _imgH = decoded.height;

      final result = _pipeline.run(decoded);

      setState(() {
        _imageBytes = bytes;
        _result     = result;
        _loading    = false;
      });
    } catch (e) {
      print('Pipeline error: $e');
      setState(() {
        _loading = false;
        _result  = PipelineResult('Error: $e', null, 0.0);
      });
    }
  }

  @override
  void dispose() { _pipeline.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      appBar: AppBar(
        backgroundColor: Colors.deepPurple,
        title: const Text('Detect & Classify',
            style: TextStyle(color: Colors.white)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisSize: MainAxisSize.min, children: [

            if (_imageBytes != null && _imgW != null && _imgH != null)
              _BoxOverlay(
                imageBytes:  _imageBytes!,
                result:      _result,
                displaySize: 300,
                imageWidth:  _imgW!,
                imageHeight: _imgH!,
              )
            else
              Container(
                height: 300,
                alignment: Alignment.center,
                child: const Icon(Icons.image, size: 80, color: Colors.white24),
              ),

            const SizedBox(height: 24),

            if (_loading)
              const CircularProgressIndicator(color: Colors.deepPurpleAccent)
            else if (_result != null)
              _ResultCard(result: _result!),

            const SizedBox(height: 24),

            ElevatedButton.icon(
              onPressed: _loading ? null : _pickAndRun,
              icon: const Icon(Icons.refresh),
              label: const Text('Try Another'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.deepPurple,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(
                    vertical: 14, horizontal: 24),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ]),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
// Bounding-box overlay
// ─────────────────────────────────────────────
class _BoxOverlay extends StatelessWidget {
  final Uint8List    imageBytes;
  final PipelineResult? result;
  final double       displaySize;
  final int          imageWidth;
  final int          imageHeight;

  const _BoxOverlay({
    required this.imageBytes,
    required this.result,
    required this.displaySize,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width:  displaySize,
      height: displaySize,
      child: Stack(children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Image.memory(imageBytes,
              width: displaySize, height: displaySize, fit: BoxFit.contain),
        ),
        if (result?.box != null)
          CustomPaint(
            size: Size(displaySize, displaySize),
            painter: _BoxPainter(result!.box!, imageWidth, imageHeight),
          ),
      ]),
    );
  }
}

class _BoxPainter extends CustomPainter {
  final Rect box;
  final int  imgW;
  final int  imgH;
  _BoxPainter(this.box, this.imgW, this.imgH);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color      = Colors.greenAccent
      ..style      = PaintingStyle.stroke
      ..strokeWidth = 3;

    canvas.drawRect(
      Rect.fromLTWH(
        box.left  / imgW * size.width,
        box.top   / imgH * size.height,
        box.width / imgW * size.width,
        box.height/ imgH * size.height,
      ),
      paint,
    );
  }

  @override
  bool shouldRepaint(_BoxPainter old) => old.box != box;
}

// ─────────────────────────────────────────────
// Result card
// ─────────────────────────────────────────────
class _ResultCard extends StatelessWidget {
  final PipelineResult result;
  const _ResultCard({required this.result});

  @override
  Widget build(BuildContext context) {
    final detected = result.box != null;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: BoxDecoration(
        color: detected ? Colors.deepPurple.shade800 : Colors.red.shade900,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(children: [
        if (detected)
          Text(
            'Sign detected  (${(result.detectorConf * 100).toStringAsFixed(1)}%)',
            style: const TextStyle(color: Colors.white70, fontSize: 13),
          ),
        const SizedBox(height: 6),
        Text(
          result.label,
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
      ]),
    );
  }
}

// ─────────────────────────────────────────────
// Screen 2 — Live camera
// ─────────────────────────────────────────────
class LiveCameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const LiveCameraScreen({super.key, required this.cameras});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  CameraController? _cameraController;
  final _pipeline  = SignPipeline();
  String _liveLabel = 'Initialising…';
  double _detConf   = 0.0;
  bool   _isRunning = false;

  @override
  void initState() {
    super.initState();
    _pipeline.load().then((_) => _startCamera());
  }

  Future<void> _startCamera() async {
    if (widget.cameras.isEmpty) {
      setState(() => _liveLabel = 'No cameras found');
      return;
    }
    _cameraController = CameraController(
      widget.cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    await _cameraController!.initialize();
    if (!mounted) return;
    await _cameraController!.startImageStream(_onCameraFrame);
    setState(() => _liveLabel = 'Point at a traffic sign');
  }

  void _onCameraFrame(CameraImage cameraImage) {
    if (_isRunning) return;
    _isRunning = true;

    img.Image? frame;
    try {
      if (cameraImage.format.group == ImageFormatGroup.jpeg) {
        frame = img.decodeJpg(
            Uint8List.fromList(cameraImage.planes[0].bytes));
      } else {
        // YUV420 → RGB
        final yPlane = cameraImage.planes[0].bytes;
        final uPlane = cameraImage.planes[1].bytes;
        final vPlane = cameraImage.planes[2].bytes;
        final w = cameraImage.width;
        final h = cameraImage.height;
        frame = img.Image(width: w, height: h);
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final yVal  = yPlane[y * w + x];
            final uvIdx = (y ~/ 2) * (w ~/ 2) + (x ~/ 2);
            final uVal  = uPlane[uvIdx] - 128;
            final vVal  = vPlane[uvIdx] - 128;
            frame!.setPixelRgb(
              x, y,
              (yVal + 1.402 * vVal).clamp(0, 255).toInt(),
              (yVal - 0.344136 * uVal - 0.714136 * vVal)
                  .clamp(0, 255).toInt(),
              (yVal + 1.772 * uVal).clamp(0, 255).toInt(),
            );
          }
        }
      }
    } catch (_) { _isRunning = false; return; }

    if (frame == null) { _isRunning = false; return; }

    final result = _pipeline.run(frame);
    if (mounted) {
      setState(() {
        _liveLabel = result.label;
        _detConf   = result.detectorConf;
      });
    }
    _isRunning = false;
  }

  @override
  void dispose() {
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _pipeline.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isReady  = _cameraController?.value.isInitialized ?? false;
    final detected = _detConf >= kConfThreshold;

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.deepPurple,
        title: const Text('Live Detection',
            style: TextStyle(color: Colors.white)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Stack(children: [
        if (isReady)
          SizedBox.expand(child: CameraPreview(_cameraController!))
        else
          const Center(child: CircularProgressIndicator(
              color: Colors.deepPurpleAccent)),

        Positioned(
          bottom: 0, left: 0, right: 0,
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 20),
            color: Colors.black.withOpacity(0.70),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              if (detected)
                Text(
                  'Sign detected  '
                  '(${(_detConf * 100).toStringAsFixed(1)}%)',
                  style: const TextStyle(
                      color: Colors.greenAccent, fontSize: 13),
                ),
              const SizedBox(height: 4),
              Text(
                _liveLabel,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: detected ? Colors.white : Colors.white54,
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ]),
          ),
        ),
      ]),
    );
  }
}