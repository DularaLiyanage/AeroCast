import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';

import '../data/forecast_service.dart';
import '../utils/forecast_utils.dart';
import '../widgets/forecast_chart.dart';
import '../widgets/xai_section.dart'; 
import '../widgets/pollutant_selector.dart';
import '../widgets/policy_section.dart';
import '../widgets/hero_card.dart';
import '../../risk_scoring/utils/constants.dart'; // Added AppColors

class ForecastScreen extends StatefulWidget {
  const ForecastScreen({super.key});

  @override
  State<ForecastScreen> createState() => _ForecastScreenState();
}

class _ForecastScreenState extends State<ForecastScreen> {
  final ForecastService _api = ForecastService(); 
  
  String selectedLocation = "baththaramulla";
  String selectedPollutant = "PM2 5 Conc";
  int selectedHourIndex = 0; // <--- NEW STATE VARIABLE
  
  Map<String, dynamic>? forecastData;
  bool isLoading = false;
  bool isError = false;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() { isLoading = true; isError = false; });
    try {
      final data = await _api.fetchForecast(selectedLocation);
      setState(() {
        forecastData = data;
        selectedHourIndex = 0; // Reset slider on new data load
        if (forecastData != null && !forecastData!.containsKey(selectedPollutant)) {
          selectedPollutant = forecastData!.keys.firstWhere((k) => !k.endsWith('_xai'));
        }
      });
    } catch (e) {
      print("Error loading data: $e");
      setState(() => isError = true);
    } finally {
      setState(() => isLoading = false);
    }
  }

  // Helper to get formatted time string for the card
  String _getSelectedTimeText() {
    DateTime now = DateTime.now();
    DateTime tomorrowStart = DateTime(now.year, now.month, now.day).add(const Duration(days: 1));
    DateTime selectedTime = tomorrowStart.add(Duration(hours: selectedHourIndex));
    return DateFormat('EEEE, h:00 a').format(selectedTime); // e.g. "Wednesday, 2:00 PM"
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text("Forecast", style: GoogleFonts.dmSans(fontWeight: FontWeight.w800, color: AppColors.primaryText)),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: AppColors.primaryText),
      ),
      body: isLoading 
          ? const Center(child: CircularProgressIndicator())
          : isError 
              ? Center(child: ElevatedButton(onPressed: _loadData, child: const Text("Retry")))
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildLocationDropdown(),
                      const SizedBox(height: 20),

                      PollutantSelector(
                        forecastData: forecastData,
                        selectedPollutant: selectedPollutant,
                        onPollutantChanged: (newValue) {
                          setState(() {
                            selectedPollutant = newValue;
                            selectedHourIndex = 0; // Reset slider when pollutant changes
                          });
                        },
                      ),
                      
                      const SizedBox(height: 30),

                      // --- HERO CARD (UPDATED) ---
                      if (forecastData != null)
                        HeroCard(
                          // Get value at specific hour index
                          value: (forecastData![selectedPollutant][selectedHourIndex] as num).toDouble(), 
                          pollutant: selectedPollutant,
                          time: _getSelectedTimeText(), // Dynamic Time
                        ),

                      const SizedBox(height: 30),

                      const Text("24-Hour Trend", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),

                      // --- CHART (UPDATED) ---
                      if (forecastData != null)
                        ForecastChart(
                          values: forecastData![selectedPollutant], 
                          pollutant: selectedPollutant,
                          selectedHourIndex: selectedHourIndex, // Pass Index
                          onHourChanged: (newIndex) {           // Receive Update
                            setState(() {
                              selectedHourIndex = newIndex;
                            });
                          },
                        ),

                      const SizedBox(height: 30),

                      if (forecastData != null)
                        PolicySection(
                          values: forecastData![selectedPollutant],
                          pollutant: selectedPollutant,
                        ),

                      const SizedBox(height: 30),

                      if (forecastData != null && forecastData!.containsKey("${selectedPollutant}_xai"))
                         XaiSection(rawXaiData: forecastData!["${selectedPollutant}_xai"]),
                         
                      const SizedBox(height: 50),
                    ],
                  ),
                ),
    );
  }

  Widget _buildLocationDropdown() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.grey[50],
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey[200]!),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: selectedLocation,
          icon: const Icon(Icons.keyboard_arrow_down_rounded, color: Colors.black54),
          isExpanded: true,
          items: ["baththaramulla", "kandy"].map((String value) {
            return DropdownMenuItem<String>(
              value: value,
              child: Text(
                value.toUpperCase(), 
                style: GoogleFonts.dmSans(fontWeight: FontWeight.w700, fontSize: 14, color: Colors.black87),
              ),
            );
          }).toList(),
          onChanged: (newValue) {
            if (newValue != null) {
              setState(() => selectedLocation = newValue);
              _loadData();
            }
          },
        ),
      ),
    );
  }
}