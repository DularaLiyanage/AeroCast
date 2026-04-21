import 'package:flutter/foundation.dart';
import '../models/aqi_prediction.dart';
import '../services/api_service.dart';
import '../../../services/notification_service.dart';

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
      
      // Proactively trigger local notification if an alert is present
      if (_predictions.isNotEmpty) {
        final current = _predictions.first;
        if (current.healthAlert != null) {
          final title = current.healthAlert!['title'] ?? 'AQI Health Alert';
          final message = current.healthAlert!['message'] ?? 'Please check the app for details.';
          
          // Use Future.delayed to ensure it's not blocking the UI frame
          Future.microtask(() async {
            await NotificationService.showAqiAlert(title: title, message: message, id: 101);
          });
        }
      }
    } catch (e) {
      _error = e.toString();
      _predictions = [];
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}
