import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/aqi_prediction.dart';
import '../utils/constants.dart';

class DetailedReportScreen extends StatelessWidget {
  final List<AqiPrediction> predictions;
  final String location;

  const DetailedReportScreen({
    super.key,
    required this.predictions,
    required this.location,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text(
          "$location Report",
          style: GoogleFonts.poppins(
              fontWeight: FontWeight.w600, color: AppColors.primaryText),
        ),
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new,
              color: AppColors.primaryText),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: ListView.separated(
        padding: const EdgeInsets.all(16),
        physics: const BouncingScrollPhysics(),
        itemCount: predictions.length,
        separatorBuilder: (context, index) => const SizedBox(height: 12),
        itemBuilder: (context, index) {
          final p = predictions[index];
          final isCurrent = index == 0;

          return Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: isCurrent
                  ? Border.all(color: AppColors.getColor(p.colour), width: 2)
                  : Border.all(color: Colors.grey.withOpacity(0.1)),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.02),
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                )
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          p.time,
                          style: GoogleFonts.poppins(
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                            color: AppColors.primaryText,
                          ),
                        ),
                        Text(
                          p.date,
                          style: GoogleFonts.poppins(
                            fontSize: 12,
                            color: Colors.grey,
                          ),
                        ),
                      ],
                    ),
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color:
                                AppColors.getColor(p.colour).withOpacity(0.1),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            "${p.aqi.toInt()} AQI",
                            style: GoogleFonts.poppins(
                              fontWeight: FontWeight.bold,
                              color: AppColors.getColor(p.colour),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    _buildStat("Min", p.min.toStringAsFixed(1)),
                    const SizedBox(width: 20),
                    _buildStat("Max", p.max.toStringAsFixed(1)),
                    const Spacer(),
                    Text(
                      p.riskLevel,
                      style: GoogleFonts.poppins(
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                          color: AppColors.secondaryText),
                    )
                  ],
                ),
                if (p.reasoning.isNotEmpty) ...[
                  const Divider(height: 24),
                  Text(
                    p.reasoning,
                    style: GoogleFonts.poppins(
                      fontSize: 13,
                      color: Colors.grey[700],
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ]
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildStat(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label.toUpperCase(),
          style: GoogleFonts.poppins(
              fontSize: 10, color: Colors.grey, fontWeight: FontWeight.w600),
        ),
        Text(
          value,
          style: GoogleFonts.poppins(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: AppColors.primaryText),
        ),
      ],
    );
  }
}
