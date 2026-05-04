import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../utils/forecast_utils.dart';
import '../../risk_scoring/utils/constants.dart';

class PolicySection extends StatelessWidget {
  final List<dynamic> values;
  final String pollutant;
  final Map<String, dynamic>? xaiData;

  const PolicySection({
    super.key,
    required this.values,
    required this.pollutant,
    this.xaiData,
  });

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

    // 2. Get Specific Actions — safe lookup, no crash if pollutant missing
    final actions = ForecastUtils.policyActions[pollutant];
    if (actions == null || actions.isEmpty) return const SizedBox.shrink();

    // 3. Derive XAI context (dominant category + tip)
    String? dominantCategory;
    if (xaiData != null && xaiData!.isNotEmpty) {
      dominantCategory = ForecastUtils.getDominantXaiCategory(xaiData!);
    }
    final contextTip = dominantCategory != null
        ? ForecastUtils.getXaiContextualTip(dominantCategory)
        : null;
    final contextColor =
        ForecastUtils.xaiCategoryColors[dominantCategory] ?? Colors.blueGrey;
    final contextIcon =
        ForecastUtils.xaiCategoryIcons[dominantCategory] ?? Icons.info_outline;

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
          const SizedBox(height: 16),

          // XAI context note
          if (contextTip != null) ...[
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: contextColor.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(10),
                border: Border(left: BorderSide(color: contextColor, width: 3)),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(contextIcon, size: 14, color: contextColor),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      contextTip,
                      style: TextStyle(
                        fontSize: 12,
                        color: contextColor,
                        fontWeight: FontWeight.w500,
                        height: 1.4,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],

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