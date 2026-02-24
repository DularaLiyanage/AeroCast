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

  // ... [Paste your policyActions Map here] ...
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
}