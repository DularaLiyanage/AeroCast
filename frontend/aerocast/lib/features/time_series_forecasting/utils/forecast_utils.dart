import 'package:flutter/material.dart';

class ForecastUtils {
  static const Map<String, String> pollutantLabels = {
    "PM2 5 Conc": "PM2.5",
    "PM10 Conc": "PM10",
    "NO2 Conc": "NO₂",
    "SO2 Conc": "SO₂",
    "O3 Conc": "O₃",
    "CO Conc": "CO",
  };

  static Color getStatusColor(double value, String pollutant) {
    // 1. PM2.5 (µg/m³) - CEA Standard
    if (pollutant == "PM2 5 Conc") {
      if (value <= 25.0) return const Color(0xFF00E400); // Good (Green)
      if (value <= 50.0) return const Color(0xFFFFD700); // Moderate (Yellow)
      if (value <= 75.0) return const Color(0xFFFF7E00); // Slightly Unhealthy (Orange)
      if (value <= 150.0) return const Color(0xFFFF0000); // Unhealthy (Red)
      if (value <= 250.0) return const Color(0xFF8F3F97); // Very Unhealthy (Purple)
      return const Color(0xFF7E0023); // Hazardous (Maroon)
    }

    // 2. PM10 (µg/m³) - CEA Standard
    if (pollutant == "PM10 Conc") {
      if (value <= 50.0) return const Color(0xFF00E400);
      if (value <= 100.0) return const Color(0xFFFFD700);
      if (value <= 150.0) return const Color(0xFFFF7E00);
      if (value <= 275.0) return const Color(0xFFFF0000);
      if (value <= 450.0) return const Color(0xFF8F3F97);
      return const Color(0xFF7E0023);
    }

    // 3. NO2 (ppb) - CEA Standard
    if (pollutant == "NO2 Conc") {
      if (value <= 65.0) return const Color(0xFF00E400);
      if (value <= 130.0) return const Color(0xFFFFD700);
      if (value <= 350.0) return const Color(0xFFFF7E00);
      if (value <= 650.0) return const Color(0xFFFF0000);
      if (value <= 1250.0) return const Color(0xFF8F3F97);
      return const Color(0xFF7E0023);
    }
    
    // 4. SO2 (ppb) - CEA Standard
    if (pollutant == "SO2 Conc") {
      if (value <= 15.0) return const Color(0xFF00E400); // CEA is very strict on SO2!
      if (value <= 30.0) return const Color(0xFFFFD700);
      if (value <= 80.0) return const Color(0xFFFF7E00);
      if (value <= 250.0) return const Color(0xFFFF0000);
      if (value <= 600.0) return const Color(0xFF8F3F97);
      return const Color(0xFF7E0023);
    }

    // 5. OZONE (ppb) - CEA Standard
    if (pollutant == "O3 Conc") {
      if (value <= 50.0) return const Color(0xFF00E400);
      if (value <= 100.0) return const Color(0xFFFFD700);
      if (value <= 200.0) return const Color(0xFFFF7E00);
      if (value <= 300.0) return const Color(0xFFFF0000);
      if (value <= 400.0) return const Color(0xFF8F3F97);
      return const Color(0xFF7E0023);
    }

    // 6. CO (ppb) - CEA Standard
    // Note: If your data is in ppm, you must multiply by 1000 before sending here!
    if (pollutant == "CO Conc") { 
      if (value <= 2250.0) return const Color(0xFF00E400);
      if (value <= 4500.0) return const Color(0xFFFFD700);
      if (value <= 9000.0) return const Color(0xFFFF7E00);
      if (value <= 15000.0) return const Color(0xFFFF0000);
      if (value <= 30000.0) return const Color(0xFF8F3F97);
      return const Color(0xFF7E0023);
    }

    // Fallback default
    if (value <= 50.0) return const Color(0xFF00E400);
    return const Color(0xFFFF0000);
  }

  static String getStatusText(double value, String pollutant) {
    Color color = getStatusColor(value, pollutant);
    
    if (color == const Color(0xFF00E400)) return "Good";
    if (color == const Color(0xFFFFD700)) return "Moderate";
    if (color == const Color(0xFFFF7E00)) return "Slightly Unhealthy"; // Official CEA Term
    if (color == const Color(0xFFFF0000)) return "Unhealthy";
    if (color == const Color(0xFF8F3F97)) return "Very Unhealthy";
    return "Hazardous";
  }
  
  static IconData getIconData(String iconName) {
    switch (iconName) {
      case 'construction': return Icons.construction;
      case 'cleaning_services': return Icons.cleaning_services;
      case 'local_shipping': return Icons.local_shipping;
      case 'traffic': return Icons.traffic;
      case 'factory': return Icons.factory;
      case 'local_fire_department': return Icons.local_fire_department;
      case 'local_gas_station': return Icons.local_gas_station;
      case 'warning': return Icons.warning_amber_rounded;
      case 'wb_sunny': return Icons.wb_sunny;
      default: return Icons.admin_panel_settings;
    }
  }
  
  static String getUnit(String pollutant) {
    if (pollutant == "PM2 5 Conc" || pollutant == "PM10 Conc") {
      return "µg/m³";
    }
    return "ppb";
  }

  static final Map<String, List<Map<String, String>>> policyActions = {
    "PM10 Conc": [
      {"icon": "construction", "text": "Mandate water sprinkling at construction sites to reduce road dust."},
      {"icon": "cleaning_services", "text": "Deploy street sweepers to control wind-blown dust."},
    ],
    "NO2 Conc": [
      {"icon": "local_shipping", "text": "Restrict heavy vehicle entry (Lorries/Buses) during peak hours."},
      {"icon": "traffic", "text": "Minimize vehicle idling at junctions to reduce combustion emissions."},
    ],
    "PM2 5 Conc": [
      {"icon": "local_fire_department", "text": "Enforce ban on biomass/waste burning."},
      {"icon": "factory", "text": "Inspect industrial processes and vehicle emissions."},
    ],
    "SO2 Conc": [
      {"icon": "local_gas_station", "text": "Inspect industrial fuel quality (sulfur content)."},
      {"icon": "warning", "text": "Issue advisory to power plants using fossil fuels."},
    ],
    "O3 Conc": [
      {"icon": "wb_sunny", "text": "Limit use of volatile organic compounds (paints/solvents) during midday."},
      {"icon": "directions_car", "text": "Manage traffic flow to reduce NOx precursors."},
    ],
    "CO Conc": [
      {"icon": "directions_car", "text": "Check for incomplete combustion in vehicle fleets."},
      {"icon": "factory", "text": "Ensure proper ventilation in industrial heating zones."},
    ]
  };

  // ── XAI Utilities ──────────────────────────────────────────────────────────

  static const Map<String, String> xaiFriendlyNames = {
    "AT": "Temperature",
    "RH": "Humidity",
    "BP": "Atmospheric Pressure",
    "Rain Gauge": "Rainfall",
    "Solar Radiation": "Solar Intensity",
    "WD_sin": "Wind Direction",
    "WD_cos": "Wind Direction",
    "Heat_Humidity_Interaction": "Heat Index",
    "time_idx": "Long-term Trend",
    "traffic_hour": "Traffic Patterns",
    "is_weekend": "Weekend Effect",
    "is_holiday": "Holiday Effect",
    "hour": "Time of Day",
    "month": "Seasonal Effect",
    "monsoon_phase_First Inter-monsoon": "Monsoon Season",
    "monsoon_phase_Northeast Monsoon": "Monsoon Season",
    "monsoon_phase_Second Inter-monsoon": "Monsoon Season",
    "monsoon_phase_Southwest Monsoon": "Monsoon Season",
    "sarimax_pred_scaled": "Historical Baseline",
    "PM2 5 Conc_lag24": "Past Pollution Levels",
    "PM2 5 Conc_rolling24_mean": "Past Pollution Levels",
    "PM10 Conc_lag24": "Past Dust Levels",
    "PM10 Conc_rolling24_mean": "Past Dust Levels",
  };

  static const Map<String, String> xaiDriverCategories = {
    "Temperature": "Weather",
    "Humidity": "Weather",
    "Atmospheric Pressure": "Weather",
    "Rainfall": "Weather",
    "Solar Intensity": "Weather",
    "Wind Direction": "Weather",
    "Heat Index": "Weather",
    "Long-term Trend": "Time",
    "Traffic Patterns": "Time",
    "Weekend Effect": "Time",
    "Holiday Effect": "Time",
    "Time of Day": "Time",
    "Seasonal Effect": "Season",
    "Monsoon Season": "Season",
    "Historical Baseline": "Historical",
    "Past Pollution Levels": "Historical",
    "Past Dust Levels": "Historical",
  };

  static const Map<String, Color> xaiCategoryColors = {
    "Weather": Color(0xFF1E88E5),
    "Time": Color(0xFF8E24AA),
    "Season": Color(0xFF43A047),
    "Historical": Color(0xFFF4511E),
  };

  static const Map<String, IconData> xaiCategoryIcons = {
    "Weather": Icons.wb_cloudy_outlined,
    "Time": Icons.schedule_rounded,
    "Season": Icons.eco_outlined,
    "Historical": Icons.show_chart_rounded,
  };

  // Groups raw 0–1 weights by friendly name, converts to %, sorts descending.
  static List<MapEntry<String, double>> processXaiDrivers(
      Map<String, dynamic> raw, {int topN = 5}) {
    final Map<String, double> grouped = {};
    raw.forEach((key, value) {
      final name = xaiFriendlyNames[key] ?? key.replaceAll('_', ' ');
      grouped[name] = (grouped[name] ?? 0.0) + (value as num).toDouble();
    });
    final pct = grouped.map((k, v) => MapEntry(k, v * 100.0));
    final sorted = pct.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    return sorted.length > topN ? sorted.sublist(0, topN) : sorted;
  }

  static String? getTopXaiDriver(Map<String, dynamic> raw) {
    final drivers = processXaiDrivers(raw, topN: 1);
    return drivers.isNotEmpty ? drivers.first.key : null;
  }

  // Returns category totals as percentages, e.g. {"Weather": 42.3, "Time": 31.1, ...}
  static Map<String, double> getXaiCategoryBreakdown(Map<String, dynamic> raw) {
    final Map<String, double> breakdown = {};
    raw.forEach((key, value) {
      final name = xaiFriendlyNames[key] ?? key.replaceAll('_', ' ');
      final category = xaiDriverCategories[name] ?? 'Weather';
      breakdown[category] = (breakdown[category] ?? 0.0) + (value as num).toDouble();
    });
    return breakdown.map((k, v) => MapEntry(k, v * 100.0));
  }

  static String getDominantXaiCategory(Map<String, dynamic> raw) {
    final breakdown = getXaiCategoryBreakdown(raw);
    if (breakdown.isEmpty) return 'Weather';
    return breakdown.entries.reduce((a, b) => a.value > b.value ? a : b).key;
  }

  static String getXaiContextualTip(String category) {
    switch (category) {
      case 'Weather':
        return 'Wind, pressure, and temperature are the main influence. A shift in weather could quickly change these levels.';
      case 'Historical':
        return 'Levels are following a persistent trend from recent days. This is likely to continue unless conditions change significantly.';
      case 'Time':
        return 'Daily commute patterns are a key driver. Expect higher levels during rush hours (7–10 AM and 5–8 PM).';
      case 'Season':
        return 'Monsoon and seasonal patterns are the primary influence on this forecast.';
      default:
        return 'Multiple factors are contributing to this forecast.';
    }
  }
}