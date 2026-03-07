import 'package:flutter/material.dart';

class AppColors {
  // Main Theme Colors
  static const Color background = Color(0xFFF8F9FA); // Light gray background
  static const Color cardGray = Color(0xFFFFFFFF); // White/Very light gray for cards
  
  static const Color primaryBlue = Color(0xFF1E88E5);
  static const Color lightBlue = Color(0xFF64B5F6);
  static const Color darkBlue = Color(0xFF1565C0);
  static const Color accentBlue = Color(0xFF82B1FF);

  // Text Colors
  static const Color primaryText = Color(0xFF1A1A1A);
  static const Color secondaryText = Color(0xFF6E6E6E);
  
  // Legacy Gradients (Keeping variable names to prevent breaks, but mapped to blue)
  static const Color battaramullaStart = Color(0xFF1E88E5);
  static const Color battaramullaEnd = Color(0xFF64B5F6);
  
  static const Color kandyStart = Color(0xFF1565C0);
  static const Color kandyEnd = Color(0xFF1E88E5);

  // Health and Risk Colors
  static Color getColor(String colourField) {
    switch (colourField.toLowerCase()) {
      case 'green': return Colors.greenAccent.shade700;
      case 'yellow': return Colors.amber;
      case 'orange': return Colors.deepOrange;
      case 'red': return Colors.redAccent.shade700;
      case 'purple': return Colors.purple;
      case 'maroon': return const Color(0xFF800000);
      default: return Colors.blue.shade600; // Default to theme blue instead of grey
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
  static BoxShadow coloredShadow(Color color) => BoxShadow(
    color: color.withOpacity(0.2),
    blurRadius: 12,
    offset: const Offset(0, 6),
  );
}
