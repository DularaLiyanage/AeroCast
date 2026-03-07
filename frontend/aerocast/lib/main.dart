import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

// Import component screens
import 'features/time_series_forecasting/time_series_forecasting.dart';
import 'features/spatial_interpolation/spatial_interpolation.dart';
import 'features/anomaly_detection/anomaly_detection.dart';
import 'features/risk_scoring/risk_scoring.dart';
import 'features/risk_scoring/providers/aqi_provider.dart';
import 'features/risk_scoring/utils/constants.dart';

void main() {
  runApp(const AeroCastApp());
}

class AeroCastApp extends StatelessWidget {
  const AeroCastApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => AqiProvider(),
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
  const SplashScreen({super.key});

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
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 40),
              Text(
                'AeroCast',
                style: GoogleFonts.poppins(
                  fontSize: 44,
                  fontWeight: FontWeight.bold,
                  color: AppColors.primaryText,
                  height: 1.1,
                ),
              ).animate().fadeIn(duration: 800.ms).slideY(begin: 0.3, end: 0),
              const SizedBox(height: 12),
              Text(
                'Select a module to explore real-time air quality data.',
                style: GoogleFonts.poppins(
                  fontSize: 16,
                  color: AppColors.secondaryText,
                ),
              ).animate().fadeIn(delay: 200.ms).slideX(),
              const SizedBox(height: 40),
              Expanded(
                child: ListView(
                  physics: const BouncingScrollPhysics(),
                  children: [
                    _buildFeatureCard(
                      context,
                      title: 'Time Series Forecasting',
                      subtitle: '24h Pollutant Predictions',
                      startColor: const Color(0xFF3B82F6),
                      endColor: const Color(0xFF60A5FA),
                      icon: Icons.show_chart_rounded,
                      targetScreen: const ForecastScreen(),
                    )
                        .animate()
                        .fadeIn(delay: 400.ms)
                        .slideY(begin: 0.2, end: 0),
                    const SizedBox(height: 20),
                    _buildFeatureCard(
                      context,
                      title: 'Spatial Interpolation',
                      subtitle: 'High-Resolution Maps',
                      startColor: const Color(0xFF8B5CF6),
                      endColor: const Color(0xFFA78BFA),
                      icon: Icons.map_rounded,
                      targetScreen: const SpatialScreen(),
                    )
                        .animate()
                        .fadeIn(delay: 500.ms)
                        .slideY(begin: 0.2, end: 0),
                    const SizedBox(height: 20),
                    _buildFeatureCard(
                      context,
                      title: 'AQI Risk Scoring',
                      subtitle: 'Health & Risk Intelligence',
                      startColor: AppColors.battaramullaStart,
                      endColor: AppColors.battaramullaEnd,
                      icon: Icons.health_and_safety_rounded,
                      targetScreen:
                          const DashboardScreen(), // Navigates to the Risk Scoring Dashboard Screen
                    )
                        .animate()
                        .fadeIn(delay: 600.ms)
                        .slideY(begin: 0.2, end: 0),
                    const SizedBox(height: 20),
                    _buildFeatureCard(
                      context,
                      title: 'Anomaly Detection',
                      subtitle: 'Identify Pollution Events',
                      startColor: const Color(0xFFF59E0B),
                      endColor: const Color(0xFFFBBF24),
                      icon: Icons.warning_amber_rounded,
                      targetScreen: const AnomalyScreen(),
                    )
                        .animate()
                        .fadeIn(delay: 700.ms)
                        .slideY(begin: 0.2, end: 0),
                    const SizedBox(height: 40),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFeatureCard(
    BuildContext context, {
    required String title,
    required String subtitle,
    required Color startColor,
    required Color endColor,
    required IconData icon,
    required Widget targetScreen,
  }) {
    return GestureDetector(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => targetScreen),
        );
      },
      child: Container(
        height: 160,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(AppStyles.cardRadius),
          gradient: LinearGradient(
            colors: [startColor, endColor],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          boxShadow: [
            BoxShadow(
              color: startColor.withOpacity(0.4),
              blurRadius: 20,
              offset: const Offset(0, 10),
            )
          ],
        ),
        child: Stack(
          children: [
            Positioned(
              right: -10,
              bottom: -10,
              child: Icon(
                icon,
                size: 130,
                color: Colors.white.withOpacity(0.15),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Text(
                    subtitle,
                    style: GoogleFonts.poppins(
                      color: Colors.white70,
                      fontSize: 14,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    title,
                    style: GoogleFonts.poppins(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.w600,
                      height: 1.1,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
