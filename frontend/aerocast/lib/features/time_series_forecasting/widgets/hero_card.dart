import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart'; 
import '../utils/forecast_utils.dart'; 

class HeroCard extends StatelessWidget { 
  final double value;
  final String pollutant;
  final String time;

  const HeroCard({
    Key? key,
    required this.value,
    required this.pollutant,
    required this.time,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final statusColor = ForecastUtils.getStatusColor(value, pollutant);
    final statusText = ForecastUtils.getStatusText(value, pollutant);

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      padding: const EdgeInsets.all(28),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(32),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 30,
            offset: const Offset(0, 15),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // --- FIX STARTS HERE ---
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              // 1. Wrap the text in Expanded so it shrinks if needed
              Expanded(
                child: Text(
                  "Forecast · $time", 
                  style: TextStyle(color: Colors.grey[600]),
                  overflow: TextOverflow.ellipsis, // Prevents crash if too long
                  maxLines: 1,
                ),
              ),
              
              const SizedBox(width: 8), // Gap between text and badge

              // 2. The Badge (keeps its size)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                decoration: BoxDecoration(
                  color: statusColor.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  statusText,
                  style: TextStyle(
                    color: statusColor,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          // --- FIX ENDS HERE ---
          
          const SizedBox(height: 30),
          Center(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  value.toStringAsFixed(1),
                  style: GoogleFonts.dmSans(
                    fontSize: 72,
                    fontWeight: FontWeight.w800,
                    height: 1,
                    color: Colors.black87,
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(bottom: 12, left: 8),
                  child: Text(
                    "µg/m³",
                    style: TextStyle(color: Colors.grey[400], fontSize: 18),
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