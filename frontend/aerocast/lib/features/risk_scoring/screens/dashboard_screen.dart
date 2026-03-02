import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'detailed_report_screen.dart';
import '../providers/aqi_provider.dart';
import '../utils/constants.dart';
import '../widgets/aqi_chart.dart';
import '../widgets/custom_card.dart';

class DashboardScreen extends StatefulWidget {
  final String location;

  const DashboardScreen({super.key, required this.location});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  @override
  void initState() {
    super.initState();
    // Fetch data after the first frame
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<AqiProvider>(context, listen: false)
          .loadPredictions(widget.location);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text(widget.location,
            style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.primaryText,
      ),
      body: Consumer<AqiProvider>(
        builder: (context, provider, child) {
          if (provider.isLoading) {
            return const Center(child: CircularProgressIndicator());
          }

          if (provider.error != null) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline,
                      size: 48, color: Colors.redAccent),
                  const SizedBox(height: 16),
                  Text('Error loading data',
                      style: GoogleFonts.poppins(fontSize: 18)),
                  Text(provider.error!,
                      style: GoogleFonts.poppins(color: Colors.grey),
                      textAlign: TextAlign.center),
                  const SizedBox(height: 24),
                  ElevatedButton(
                    onPressed: () => provider.loadPredictions(widget.location),
                    child: const Text('Try Again'),
                  )
                ],
              ),
            );
          }

          if (provider.predictions.isEmpty) {
            return const Center(child: Text("No data available."));
          }

          // Data is ready
          final current = provider.predictions.first;
          final predictions = provider.predictions;

          return SingleChildScrollView(
            physics: const BouncingScrollPhysics(),
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 0. Health Alert Banner
                  if (current.healthAlert != null)
                    Builder(builder: (context) {
                      final alertColorName =
                          current.healthAlert!['color'] as String?;
                      final alertColor = alertColorName != null
                          ? AppColors.getColor(alertColorName)
                          : AppColors.getColor(current.colour);

                      return Container(
                        width: double.infinity,
                        margin: const EdgeInsets.only(bottom: 24),
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: alertColor.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: alertColor, width: 1.5),
                        ),
                        child: Row(
                          children: [
                            Icon(Icons.warning_amber_rounded,
                                color: alertColor, size: 32),
                            const SizedBox(width: 16),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    current.healthAlert!['title'] ?? 'Alert',
                                    style: GoogleFonts.poppins(
                                        fontWeight: FontWeight.bold,
                                        color: alertColor,
                                        fontSize: 16),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    current.healthAlert!['message'] ?? '',
                                    style: GoogleFonts.poppins(
                                        color: AppColors.primaryText,
                                        fontSize: 13),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ).animate().fadeIn().slideY(begin: -0.5);
                    }),

                  // 1. Hero Status Card
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: AppColors.getColor(current.colour),
                      borderRadius: BorderRadius.circular(32),
                      boxShadow: [
                        BoxShadow(
                          color: AppColors.getColor(current.colour)
                              .withOpacity(0.4),
                          blurRadius: 20,
                          offset: const Offset(0, 10),
                        )
                      ],
                    ),
                    child: Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Current Status',
                              style: GoogleFonts.poppins(
                                  color: Colors.white70, fontSize: 14),
                            ),
                            Text(
                              "${current.date} • ${current.time}",
                              style: GoogleFonts.poppins(
                                  color: Colors.white60, fontSize: 12),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          current.aqi.toInt().toString(),
                          style: GoogleFonts.poppins(
                            color: Colors.white,
                            fontSize: 72,
                            fontWeight: FontWeight.bold,
                            height: 1,
                          ),
                        ),
                        Text(
                          'AQI',
                          style: GoogleFonts.poppins(
                              color: Colors.white54, fontSize: 16),
                        ),
                        const SizedBox(height: 16),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 16, vertical: 6),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            current.riskLevel.toUpperCase(),
                            style: GoogleFonts.poppins(
                              color: Colors.white,
                              fontWeight: FontWeight.w600,
                              letterSpacing: 1.2,
                            ),
                          ),
                        ),
                      ],
                    ),
                  )
                      .animate()
                      .scale(curve: Curves.easeOutBack, duration: 600.ms),

                  const SizedBox(height: 24),

                  // 2. Reasoning Card (Moved Up)
                  CustomCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(Icons.lightbulb_outline,
                                color: AppColors.getColor(current.colour)),
                            const SizedBox(width: 8),
                            Text(
                              'Insight',
                              style: GoogleFonts.poppins(
                                  fontSize: 16, fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Text(
                          current.reasoning,
                          style: GoogleFonts.poppins(
                            color: AppColors.secondaryText,
                            fontSize: 15,
                            height: 1.5,
                          ),
                        ),
                      ],
                    ),
                  ).animate().fadeIn(delay: 200.ms).slideY(),

                  const SizedBox(height: 32),

                  // 3. Chart Section
                  Text(
                    '24-Hour Forecast',
                    style: GoogleFonts.poppins(
                        fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    height: 250,
                    padding: const EdgeInsets.all(0),
                    child: AqiChart(predictions: predictions),
                  ).animate().fadeIn(duration: 800.ms).slideX(),

                  const SizedBox(height: 24),

                  Center(
                    child: OutlinedButton.icon(
                      onPressed: () {
                        Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (_) => DetailedReportScreen(
                                    predictions: predictions,
                                    location: widget.location)));
                      },
                      icon: const Icon(Icons.list_alt),
                      label: const Text("View Detailed Report"),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: AppColors.primaryText,
                        side: BorderSide(
                            color: AppColors.primaryText.withOpacity(0.2)),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 24, vertical: 12),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(20)),
                      ),
                    ),
                  ),

                  const SizedBox(height: 32),

                  const SizedBox(height: 32),

                  const SizedBox(height: 40),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
