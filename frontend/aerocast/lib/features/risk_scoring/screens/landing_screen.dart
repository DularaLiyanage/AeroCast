import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../utils/constants.dart';
import 'dashboard_screen.dart';

class LandingScreen extends StatelessWidget {
  const LandingScreen({super.key});

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
                'Breathe\nEasy.',
                style: GoogleFonts.poppins(
                  fontSize: 48,
                  fontWeight: FontWeight.bold,
                  color: AppColors.primaryText,
                  height: 1.1,
                ),
              ).animate().fadeIn(duration: 800.ms).slideY(begin: 0.3, end: 0),
              
              const SizedBox(height: 12),
              Text(
                'Check the air quality forecast for your city.',
                style: GoogleFonts.poppins(
                  fontSize: 16,
                  color: AppColors.secondaryText,
                ),
              ).animate().fadeIn(delay: 200.ms).slideX(),
              
              const SizedBox(height: 60),
              
              Expanded(
                child: ListView(
                  children: [
                    _buildLocationCard(
                      context,
                      'Battaramulla',
                      'Urban Greenery',
                      AppColors.battaramullaStart,
                      AppColors.battaramullaEnd,
                      Icons.park_outlined,
                    ).animate().fadeIn(delay: 400.ms).scale(curve: Curves.easeOutBack),
                    
                    const SizedBox(height: 24),
                    
                    _buildLocationCard(
                      context,
                      'Kandy',
                      'Hill Capital',
                      AppColors.kandyStart,
                      AppColors.kandyEnd,
                      Icons.landscape_outlined,
                    ).animate().fadeIn(delay: 600.ms).scale(curve: Curves.easeOutBack),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLocationCard(
    BuildContext context, 
    String city, 
    String subtitle, 
    Color startColor, 
    Color endColor, 
    IconData icon
  ) {
    return GestureDetector(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => DashboardScreen(location: city)),
        );
      },
      child: Container(
        height: 180,
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
              right: -20,
              bottom: -20,
              child: Icon(
                icon,
                size: 140,
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
                      letterSpacing: 1.0,
                    ),
                  ),
                  Text(
                    city,
                    style: GoogleFonts.poppins(
                      color: Colors.white,
                      fontSize: 32,
                      fontWeight: FontWeight.w600,
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
