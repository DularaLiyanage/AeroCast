import 'package:flutter/material.dart';

class AppColors {
  static const Color background = Color(0xFFF0F4F8); // Soft grey-blue
  static const Color primaryText = Color(0xFF1A1A1A);
  static const Color secondaryText = Color(0xFF6E6E6E);
  
  static const Color battaramullaStart = Color(0xFF11998E);
  static const Color battaramullaEnd = Color(0xFF38EF7D);
  
  static const Color kandyStart = Color(0xFF2E3192);
  static const Color kandyEnd = Color(0xFF1BFFFF);

  static Color getColor(String colourField) {
    switch (colourField.toLowerCase()) {
      case 'green': return Colors.greenAccent.shade700;
      case 'yellow': return Colors.amber;
      case 'orange': return Colors.deepOrange;
      case 'red': return Colors.redAccent.shade700;
      case 'purple': return Colors.purple;
      case 'maroon': return Color(0xFF800000);
      default: return Colors.grey;
    }
  }
}

class AppStyles {
  static const double cardRadius = 24.0;
  static const BoxShadow softShadow = BoxShadow(
    color: Colors.black12,
    blurRadius: 10,
    offset: Offset(0, 4),
  );
}
