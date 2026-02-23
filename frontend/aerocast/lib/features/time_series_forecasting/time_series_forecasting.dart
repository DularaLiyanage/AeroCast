import 'package:flutter/material.dart';

class ForecastScreen extends StatelessWidget {
  const ForecastScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Component 1")),
      body: const Center(
        child: Text(
          "Air Quality Forecast Module",
          style: TextStyle(fontSize: 20),
        ),
      ),
    );
  }
}