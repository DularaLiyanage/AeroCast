import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/air_quality.dart';
import '../../../core/config.dart';

class ApiService {
  static String get baseUrl => AppConfig.baseUrl;

  Future<AirQualityData> fetchForecast(String location) async {
    // add a client‑side timeout so the UI can recover quickly instead of
    // waiting several minutes when the backend is unreachable
    final response = await http
        .post(
          Uri.parse('$baseUrl/anomaly/forecast'), // include router prefix
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'location': location}),
        )
        .timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      return AirQualityData.fromJson(jsonDecode(response.body));
    } else {
      // include the response body/message for easier debugging
      throw Exception(
          'Failed to load forecast for $location: ${response.statusCode} ${response.body}');
    }
  }
}
