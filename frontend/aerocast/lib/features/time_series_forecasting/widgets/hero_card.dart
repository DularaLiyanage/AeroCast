import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart'; 
import '../utils/forecast_utils.dart'; 
import '../../risk_scoring/utils/constants.dart';

class HeroCard extends StatelessWidget { 
  final double value;
  final String pollutant;
  final String time;

  const HeroCard({
    super.key,
    required this.value,
    required this.pollutant,
    required this.time,
  });

  @override
  Widget build(BuildContext context) {
    final statusColor = ForecastUtils.getStatusColor(value, pollutant);
    final statusText = ForecastUtils.getStatusText(value, pollutant);

    // If the status color is the fallback grey, switch to the theme's blue
    final effectiveStatusColor = statusColor == Colors.grey ? AppColors.primaryBlue : statusColor;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 0),
      padding: const EdgeInsets.all(28),
      decoration: BoxDecoration(
        color: effectiveStatusColor,
        borderRadius: BorderRadius.circular(32),
        boxShadow: [
          AppStyles.coloredShadow(effectiveStatusColor),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      "Forecast · $time", 
                      style: GoogleFonts.poppins(
                        color: Colors.white.withOpacity(0.9),
                        fontWeight: FontWeight.w500,
                        fontSize: 13,
                      ),
                      overflow: TextOverflow.ellipsis,
                      maxLines: 1,
                    ),
                    Text(
                      "${ForecastUtils.pollutantLabels[pollutant] ?? pollutant} Concentration",
                      style: GoogleFonts.poppins(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  statusText,
                  style: GoogleFonts.poppins(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 30),
          Center(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Text(
                  value.toStringAsFixed(1),
                  style: GoogleFonts.dmSans(
                    fontSize: 72,
                    fontWeight: FontWeight.w800,
                    height: 1,
                    color: Colors.white,
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(left: 8),
                  child: Text(
                    "µg/m³",
                    style: GoogleFonts.poppins(
                      color: Colors.white.withOpacity(0.9),
                      fontSize: 18,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}