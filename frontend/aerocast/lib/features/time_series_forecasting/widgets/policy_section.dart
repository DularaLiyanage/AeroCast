import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../utils/forecast_utils.dart'; // Import the utils
import '../../risk_scoring/utils/constants.dart';

class PolicySection extends StatelessWidget { // FIX: Renamed from ForecastChart
  final List<dynamic> values;
  final String pollutant;

  const PolicySection({super.key, required this.values, required this.pollutant});

  @override
  Widget build(BuildContext context) {
    // 1. Check Severity: Is it bad enough to act?
    // We look for the MAXIMUM predicted value tomorrow.
    double maxVal = values.isNotEmpty 
        ? values.map((e) => e.toDouble()).reduce((a, b) => a > b ? a : b) 
        : 0.0;
    
    // FIX: Use ForecastUtils
    Color severityColor = ForecastUtils.getStatusColor(maxVal, pollutant);
    
    // Check if urgent (Orange/Red/Purple/Maroon)
    bool isUrgent = severityColor != const Color(0xFF00E400) && // Not Green
                    severityColor != const Color(0xFFFFD700);   // Not Yellow

    if (!isUrgent) {
      // If air is Good/Moderate, show a "No Action Needed" card
      return Container(
        margin: const EdgeInsets.symmetric(horizontal: 0, vertical: 20),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.green.shade50, // Keep semantic green for good
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.green.withOpacity(0.3)),
        ),
        child: Row(
          children: [
            Icon(Icons.check_circle, color: Colors.green[700], size: 30),
            const SizedBox(width: 15),
            const Expanded(
              child: Text(
                "No regulatory restrictions required. Air quality is within compliance limits.",
                style: TextStyle(color: Colors.green, fontWeight: FontWeight.w600),
              ),
            ),
          ],
        ),
      );
    }

    // 2. Get Specific Actions
    // FIX: Use ForecastUtils.policyActions
    // Note: We need to adapt the structure slightly if your Utils structure differs, 
    // but assuming it matches the map we created earlier:
    var actions = ForecastUtils.policyActions[pollutant] ?? ForecastUtils.policyActions["default"]!;

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 20),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: AppColors.cardGray,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.red.withOpacity(0.1)),
        boxShadow: [
          BoxShadow(
            color: Colors.red.withOpacity(0.05),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.red[50], // Keep semantic red for urgent alerts
                  shape: BoxShape.circle,
                ),
                child: Icon(Icons.gavel, color: Colors.red[700], size: 24),
              ),
              const SizedBox(width: 12),
              const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Recommended Actions",
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: AppColors.primaryText),
                  ),
                  Text(
                    "Suggested interventions to lower levels",
                    style: TextStyle(fontSize: 12, color: AppColors.secondaryText),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 20),
          ...actions.map((action) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 12.0),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // FIX: Use ForecastUtils.getIconData
                  Icon(ForecastUtils.getIconData(action['icon']!), size: 20, color: AppColors.primaryBlue),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          action['title'] ?? "",
                          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: AppColors.primaryText),
                        ),
                        Text(
                          action['desc'] ?? action['text'] ?? "",
                          style: GoogleFonts.poppins(
                            fontSize: 13,
                            color: AppColors.primaryText.withOpacity(0.7),
                            height: 1.4,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }
}