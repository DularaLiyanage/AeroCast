import '../../../../core/api_service.dart';

class ForecastService {
  // CHECK THIS: Your backend logs show the path is "/forecast", not "/predict"
  // If your Python router has @router.post("/predict"), change this to "/time_forecast/predict"
  // If your Python router has @router.post("/forecast"), keep it as is.
  static const String _endpoint = "/time_forecast/forecast"; 

  Future<Map<String, dynamic>> fetchForecast(String location) async {
    try {
      print("Sending request to: $_endpoint for $location"); // Debug Print 1
      
      final response = await ApiService.post(
        _endpoint, 
        {"location": location}
      );

      print("Server Response: $response"); // Debug Print 2 (CRITICAL)

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