import '../../../../core/api_service.dart';

class ForecastResult {
  final String forecastDate;
  final Map<String, dynamic> forecast;

  ForecastResult({required this.forecastDate, required this.forecast});
}

class ForecastService {
  static const String _forecastEndpoint = "/time_forecast/forecast";
  static const String _datesEndpoint = "/time_forecast/dates";

  Future<List<String>> fetchAvailableDates() async {
    try {
      final response = await ApiService.get(_datesEndpoint);
      if (response == null || !response.containsKey('dates')) return [];
      return List<String>.from(response['dates']);
    } catch (e) {
      print("ForecastService.fetchAvailableDates error: $e");
      return [];
    }
  }

  Future<ForecastResult> fetchForecast(String location, {String? date}) async {
    try {
      final body = <String, dynamic>{"location": location};
      if (date != null) body["date"] = date;

      final response = await ApiService.post(_forecastEndpoint, body);

      if (response == null) throw Exception("Server returned null response");
      if (response.containsKey('error')) throw Exception(response['error']);
      if (response.containsKey('detail')) throw Exception("API Error: ${response['detail']}");

      if (response.containsKey('forecast')) {
        return ForecastResult(
          forecastDate: response['forecast_date'] ?? 'unknown',
          forecast: Map<String, dynamic>.from(response['forecast']),
        );
      } else {
        throw Exception("Invalid format. Available keys: ${response.keys.toList()}");
      }
    } catch (e) {
      print("ForecastService.fetchForecast error: $e");
      rethrow;
    }
  }
}
