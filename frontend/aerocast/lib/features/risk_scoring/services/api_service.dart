import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/aqi_prediction.dart';

class ApiService {
  // Use 10.0.2.2 for Android Emulator, localhost for iOS/Web, or your machine IP for physical devices.
  static const String baseUrl = 'http://10.0.2.2:8000';

  Future<List<AqiPrediction>> fetchPredictions(String location) async {
    final url = Uri.parse('$baseUrl/aqi/predict_24h');
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'location': location}),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);

        // 1. Parse Current
        final current = AqiPrediction.fromJson(data);

        // 2. Parse Forecast
        final List<dynamic> forecastJson = data['hourly_forecast'] ?? [];
        final forecast =
            forecastJson.map((json) => AqiPrediction.fromJson(json)).toList();

        // 3. Return Combined List [Current, ...Forecast]
        return [current, ...forecast];
      } else {
        throw Exception('Failed to load data: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }
}
