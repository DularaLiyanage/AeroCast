import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/air_quality.dart';

class ApiService {
  static const String baseUrl = 'http://10.0.2.2:8080'; // Target local backend (Host IP for Android Emulator)

  Future<AirQualityData> fetchForecast(String location) async {
    final response = await http.post(
      Uri.parse('$baseUrl/forecast'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'location': location}),
    );

    if (response.statusCode == 200) {
      return AirQualityData.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to load forecast for $location');
    }
  }
}
