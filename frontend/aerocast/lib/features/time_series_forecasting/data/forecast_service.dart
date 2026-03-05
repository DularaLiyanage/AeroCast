import '../../../../core/api_service.dart';

class ForecastService {
  static const String _endpoint = "/time_forecast/forecast"; 

  Future<Map<String, dynamic>> fetchForecast(String location) async {
    try {
      print("Sending request to: $_endpoint for $location");
      
      final response = await ApiService.post(
        _endpoint, 
        {"location": location}
      );

      print("Server Response: $response");

      if (response == null) {
        throw Exception("Server returned null response");
      }

      // 1. Check for specific backend errors
      if (response.containsKey('error')) {
        throw Exception(response['error']);
      }
      
      // 2. Check for FastAPI validation errors (422)
      if (response.containsKey('detail')) {
        throw Exception("API Error: ${response['detail']}");
      }

      // 3. Success Case
      if (response.containsKey('forecast')) {
        return response['forecast'];
      } else {
        // If we get here, the JSON is valid but missing the 'forecast' key
        throw Exception("Invalid format. Available keys: ${response.keys.toList()}");
      }
    } catch (e) {
      print("Forecast Service Crash: $e");
      rethrow;
    }
  }
}