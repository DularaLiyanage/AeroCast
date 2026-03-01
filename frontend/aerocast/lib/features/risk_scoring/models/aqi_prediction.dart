class AqiPrediction {
  final String time;
  final String date;
  final double aqi;
  final double min;
  final double max;
  final String riskLevel;
  final String colour;
  final String reasoning;
  final Map<String, dynamic>? healthAlert;

  AqiPrediction({
    required this.time,
    required this.date,
    required this.aqi,
    required this.min,
    required this.max,
    required this.riskLevel,
    required this.colour,
    required this.reasoning,
    this.healthAlert,
  });

  factory AqiPrediction.fromJson(Map<String, dynamic> json) {
    return AqiPrediction(
      time: json['time'] ?? '',
      date: json['date'] ?? '',
      aqi: (json['aqi'] as num).toDouble(),
      min: (json['min_aqi'] as num).toDouble(),
      max: (json['max_aqi'] as num).toDouble(),
      riskLevel: json['risk_level'] ?? 'Unknown',
      colour: json['risk_color'] ?? 'grey',
      reasoning: json['reasoning'] ?? 'No data available.',
      healthAlert: json['health_alert'],
    );
  }
}
