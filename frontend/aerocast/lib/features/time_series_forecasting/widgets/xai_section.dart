import 'package:flutter/material.dart';

class XaiSection extends StatelessWidget { // FIX: Renamed Class
  final Map<String, dynamic> rawXaiData; // FIX: Added variable

  const XaiSection({Key? key, required this.rawXaiData}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // 1. DEFINITIONS: Map Technical Names to User Friendly Names
    final Map<String, String> friendlyNames = {
      // Weather
      "AT": "Temperature",
      "RH": "Humidity",
      "BP": "Atmospheric Pressure",
      "Rain Gauge": "Rainfall",
      "Solar Radiation": "Solar Intensity",
      
      // Time & Traffic
      "time_idx": "Long-term Trend",
      "traffic_hour": "Traffic Patterns",
      "is_weekend": "Weekend Effect",
      "is_holiday": "Holiday Effect",
      "hour": "Time of Day",
      "month": "Seasonal Effect",
      
      // Complex Features
      "Heat_Humidity_Interaction": "Heat Index",
      "monsoon_phase_First Inter-monsoon": "Monsoon Season",
      "monsoon_phase_Northeast Monsoon": "Monsoon Season",
      "monsoon_phase_Second Inter-monsoon": "Monsoon Season",
      
      // Model Internals
      "sarimax_pred_scaled": "Daily Seasonal Cycle",
      "PM2 5 Conc_lag24": "Past Pollution Levels",
      "PM2 5 Conc_rolling24_mean": "Past Pollution Levels",
      "PM10 Conc_lag24": "Past Dust Levels",
      "PM10 Conc_rolling24_mean": "Past Dust Levels",
      
      // Wind 
      "WD_sin": "Wind Direction",
      "WD_cos": "Wind Direction",
    };

    // 2. LOGIC: Group and Sum the Weights
    Map<String, double> processedDrivers = {};

    rawXaiData.forEach((key, value) {
      // Ensure value is treated as double
      double weight = (value as num).toDouble();
      
      // Find the friendly name (or use the original if missing)
      String cleanName = friendlyNames[key] ?? key.replaceAll('_', ' ');

      // Add to the merge map (Summing up weights if names match)
      if (processedDrivers.containsKey(cleanName)) {
        processedDrivers[cleanName] = processedDrivers[cleanName]! + weight;
      } else {
        processedDrivers[cleanName] = weight;
      }
    });

    // 3. SORT: Highest Impact First
    var sortedEntries = processedDrivers.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    // 4. DISPLAY: Take Top 4 Only
    if (sortedEntries.length > 4) {
      sortedEntries = sortedEntries.sublist(0, 4);
    }

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 0, vertical: 20),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
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
                  color: Colors.blue[50],
                  shape: BoxShape.circle,
                ),
                child: Icon(Icons.analytics_outlined, color: Colors.blue[700], size: 20),
              ),
              const SizedBox(width: 12),
              const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Primary Drivers",
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  Text(
                    "What is causing this forecast?",
                    style: TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 24),
          if (sortedEntries.isEmpty)
             const Text("No significant drivers identified."),

          ...sortedEntries.map((entry) {
            final double value = entry.value;
            final String name = entry.key;
            
            // Choose color based on intensity logic (assuming value is % or relevant scale)
            // Adjust threshold based on your actual XAI data range (e.g., if max is 100 or 1.0)
            Color barColor = Colors.blue;
            if (value > 5.0) barColor = const Color(0xFFFF7E00); // Example thresholds
            if (value > 10.0) barColor = const Color(0xFFFF0000); 
            
            // Simple normalization for the bar width (assuming max around 15%)
            double percentage = (value / 15.0).clamp(0.0, 1.0);

            return Padding(
              padding: const EdgeInsets.only(bottom: 16.0),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        name,
                        style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
                      ),
                      Text(
                        "${value.toStringAsFixed(1)}%",
                        style: TextStyle(fontWeight: FontWeight.bold, color: barColor),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: percentage, 
                      backgroundColor: Colors.grey[100],
                      color: barColor,
                      minHeight: 6,
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
        ],
      ),
    );
  }
}