import 'package:flutter/foundation.dart';
import '../models/aqi_prediction.dart';
import '../services/api_service.dart';

class AqiProvider with ChangeNotifier {
  final ApiService _apiService = ApiService();
  
  List<AqiPrediction> _predictions = [];
  bool _isLoading = false;
  String? _error;

  List<AqiPrediction> get predictions => _predictions;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> loadPredictions(String location) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _predictions = await _apiService.fetchPredictions(location);
    } catch (e) {
      _error = e.toString();
      _predictions = [];
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}
