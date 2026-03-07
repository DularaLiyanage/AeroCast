import 'package:flutter/material.dart';
import '../utils/forecast_utils.dart'; // Import utils for labels
import '../../risk_scoring/utils/constants.dart';

class PollutantSelector extends StatelessWidget { // FIX: Renamed Class
  final Map<String, dynamic>? forecastData;
  final String selectedPollutant;
  final Function(String) onPollutantChanged; // FIX: Callback to parent

  const PollutantSelector({
    super.key, 
    required this.forecastData, 
    required this.selectedPollutant,
    required this.onPollutantChanged,
  });

  @override
  Widget build(BuildContext context) {
    if (forecastData == null) return Container();
    
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      clipBehavior: Clip.none,
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: forecastData!.keys.where((k) => !k.endsWith('_xai')).map((key) {
          bool isSelected = key == selectedPollutant;
          return Padding(
            padding: const EdgeInsets.only(right: 12),
            child: ChoiceChip(
              // FIX: Use ForecastUtils for labels
              label: Text(ForecastUtils.pollutantLabels[key] ?? key),
              selected: isSelected,
              showCheckmark: false, // Cleaner look
              selectedColor: AppColors.primaryBlue,
              backgroundColor: Colors.grey[100],
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20), side: BorderSide(color: isSelected ? AppColors.primaryBlue : Colors.transparent)),
              labelStyle: TextStyle(
                color: isSelected ? Colors.white : AppColors.primaryText,
                fontWeight: FontWeight.w600,
              ),
              // FIX: Call the parent function instead of setState
              onSelected: (_) => onPollutantChanged(key),
            ),
          );
        }).toList(),
      ),
    );
  }
}