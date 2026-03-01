import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

// Import component screens
import 'features/time_series_forecasting/time_series_forecasting.dart';
import 'features/spatial_interpolation/spatial_interpolation.dart';
import 'features/risk_scoring/risk_scoring.dart';
import 'features/anomaly_detection/anomaly_detection.dart';
import 'features/risk_scoring/providers/aqi_provider.dart';

void main() {
  runApp(const AeroCastApp());
}

class AeroCastApp extends StatelessWidget {
  const AeroCastApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => AqiProvider(),
      child: MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'AeroCast',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          scaffoldBackgroundColor: Colors.grey[100],
        ),
        home: const SplashScreen(),
      ),
    );
  }
}

////////////////////////////////////////////////////////
/// SPLASH / LOADING SCREEN
////////////////////////////////////////////////////////

class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _loadApp();
  }

  Future<void> _loadApp() async {
    await Future.delayed(const Duration(seconds: 2));

    if (!mounted) return;

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const HomeScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const [
            Icon(Icons.air, size: 80, color: Colors.blue),
            SizedBox(height: 20),
            Text(
              "AeroCast",
              style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),
            CircularProgressIndicator(),
          ],
        ),
      ),
    );
  }
}

////////////////////////////////////////////////////////
/// HOME SCREEN WITH 4 COMPONENT BUTTONS
////////////////////////////////////////////////////////

class HomeScreen extends StatelessWidget {
  const HomeScreen({Key? key}) : super(key: key);

  Widget _buildButton(BuildContext context, String title, Widget screen) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10.0),
      child: SizedBox(
        width: double.infinity,
        height: 55,
        child: ElevatedButton(
          onPressed: () {
            Navigator.push(context, MaterialPageRoute(builder: (_) => screen));
          },
          child: Text(title, style: const TextStyle(fontSize: 16)),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("AeroCast Dashboard"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            const SizedBox(height: 30),

            _buildButton(context, "Forecast", const ForecastScreen()),

            _buildButton(context, "Spatial", const SpatialScreen()),

            _buildButton(context, "AQI", const LandingScreen()),

            _buildButton(context, "Anomaly", const AnomalyScreen()),
          ],
        ),
      ),
    );
  }
}
