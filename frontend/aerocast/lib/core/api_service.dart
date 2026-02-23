import 'dart:convert';
import 'package:http/http.dart' as http;
import 'config.dart';

class ApiService {

  static Future<dynamic> get(String endpoint) async {
    final response = await http.get(
      Uri.parse("${AppConfig.baseUrl}$endpoint"),
    );

    return jsonDecode(response.body);
  }

  static Future<dynamic> post(String endpoint, Map<String, dynamic> body) async {
    final response = await http.post(
      Uri.parse("${AppConfig.baseUrl}$endpoint"),
      headers: {
        "Content-Type": "application/json",
      },
      body: jsonEncode(body),
    );

    return jsonDecode(response.body);
  }
}