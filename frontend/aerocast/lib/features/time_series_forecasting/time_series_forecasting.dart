import 'package:flutter/material.dart';
import '../../core/api_service.dart';

class ForecastScreen extends StatefulWidget {
  const ForecastScreen({Key? key}) : super(key: key);

  @override
  State<ForecastScreen> createState() => _ForecastScreenState();
}

class _ForecastScreenState extends State<ForecastScreen> {

  String result = "No prediction yet";
  bool isLoading = false;

  Future<void> callPrediction() async {
    setState(() {
      isLoading = true;
    });

    final response = await ApiService.post(
      "/time_forecast/predict",
      {
        "temperature": 30,
        "humidity": 70
      },
    );

    setState(() {
      result = "Prediction: ${response['prediction']}";
      isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Component 1")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [

            Text(result, style: const TextStyle(fontSize: 20)),

            const SizedBox(height: 20),

            isLoading
                ? const CircularProgressIndicator()
                : ElevatedButton(
                    onPressed: callPrediction,
                    child: const Text("Get Prediction"),
                  ),
          ],
        ),
      ),
    );
  }
}