import 'package:flutter/material.dart';

// class AirQualityData {
class AirQualityData {
  final Map<String, List<double>> pollutantForecasts;
  final int? currentAqi;
  final String location;
  final Map<String, List<String>> reasons;
  final Map<String, double> thresholds;
  final Map<String, List<String>> safetyTips;

  AirQualityData({
    required this.pollutantForecasts,
    this.currentAqi,
    required this.location,
    required this.reasons,
    required this.thresholds,
    required this.safetyTips,
  });

  factory AirQualityData.fromJson(Map<String, dynamic> json) {
    var rawForecast = json['forecast_7_days'] as Map<String, dynamic>;
    Map<String, List<double>> forecasts = {};
    rawForecast.forEach((key, value) {
      forecasts[key] = List<double>.from(value.map((e) => e.toDouble()));
    });

    return AirQualityData(
      pollutantForecasts: forecasts,
      currentAqi: json['current_aqi'],
      location: json['location'],
      reasons: Map<String, List<String>>.from(
        json['reasons'].map((k, v) => MapEntry(k, List<String>.from(v))),
      ),
      thresholds: Map<String, double>.from(
        json['thresholds'].map((k, v) => MapEntry(k, v.toDouble())),
      ),
      safetyTips: Map<String, List<String>>.from(
        json['safety_tips'].map((k, v) => MapEntry(k, List<String>.from(v))),
      ),
    );
  }

  double getCurrentValue(String pollutant) {
    return pollutantForecasts[pollutant]?.first ?? 0.0;
  }
}
