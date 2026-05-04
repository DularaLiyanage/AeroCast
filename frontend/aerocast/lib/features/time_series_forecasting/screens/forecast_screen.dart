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
import '../../risk_scoring/utils/constants.dart';

class ForecastScreen extends StatefulWidget {
  const ForecastScreen({super.key});

  @override
  State<ForecastScreen> createState() => _ForecastScreenState();
}

class _ForecastScreenState extends State<ForecastScreen> {
  final ForecastService _api = ForecastService();

  String selectedLocation = "baththaramulla";
  String selectedPollutant = "PM2 5 Conc";
  int selectedHourIndex = 0;
  String? selectedDate; // null = latest
  String? forecastDate; // actual date returned by API

  List<String> availableDates = [];
  Map<String, dynamic>? forecastData;
  bool isLoading = false;
  bool isError = false;

  @override
  void initState() {
    super.initState();
    _loadDates();
  }

  Future<void> _loadDates() async {
    final dates = await _api.fetchAvailableDates();
    setState(() {
      availableDates = dates;
      selectedDate = dates.isNotEmpty ? dates.first : null;
    });
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() { isLoading = true; isError = false; });
    try {
      final result = await _api.fetchForecast(selectedLocation, date: selectedDate);
      setState(() {
        forecastData = result.forecast;
        forecastDate = result.forecastDate;
        selectedHourIndex = 0;
        if (forecastData != null && !forecastData!.containsKey(selectedPollutant)) {
          selectedPollutant = forecastData!.keys.firstWhere((k) => !k.endsWith('_xai'));
        }
      });
    } catch (e) {
      setState(() => isError = true);
    } finally {
      setState(() => isLoading = false);
    }
  }

  String _getSelectedTimeText() {
    DateTime base;
    if (forecastDate != null && forecastDate != 'unknown') {
      base = DateTime.parse(forecastDate!);
    } else {
      base = DateTime.now().add(const Duration(days: 1));
      base = DateTime(base.year, base.month, base.day);
    }
    final selectedTime = base.add(Duration(hours: selectedHourIndex));
    return DateFormat('EEEE, h:00 a').format(selectedTime);
  }

  Map<String, dynamic>? _rawXaiData() {
    if (forecastData == null) return null;
    final key = "${selectedPollutant}_xai";
    final raw = forecastData![key];
    if (raw == null) return null;
    return Map<String, dynamic>.from(raw);
  }

  String? _topXaiDriver() {
    final raw = _rawXaiData();
    if (raw == null || raw.isEmpty) return null;
    return ForecastUtils.getTopXaiDriver(raw);
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
                      const SizedBox(height: 12),
                      _buildDateButton(),
                      const SizedBox(height: 20),

                      PollutantSelector(
                        forecastData: forecastData,
                        selectedPollutant: selectedPollutant,
                        onPollutantChanged: (newValue) {
                          setState(() {
                            selectedPollutant = newValue;
                            selectedHourIndex = 0;
                          });
                        },
                      ),

                      const SizedBox(height: 30),

                      if (forecastData != null)
                        HeroCard(
                          value: (forecastData![selectedPollutant][selectedHourIndex] as num).toDouble(),
                          pollutant: selectedPollutant,
                          time: _getSelectedTimeText(),
                          topDriver: _topXaiDriver(),
                        ),

                      const SizedBox(height: 30),

                      const Text("24-Hour Trend", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),

                      if (forecastData != null)
                        ForecastChart(
                          values: forecastData![selectedPollutant],
                          pollutant: selectedPollutant,
                          selectedHourIndex: selectedHourIndex,
                          onHourChanged: (newIndex) {
                            setState(() => selectedHourIndex = newIndex);
                          },
                        ),

                      const SizedBox(height: 30),

                      if (forecastData != null)
                        PolicySection(
                          values: forecastData![selectedPollutant],
                          pollutant: selectedPollutant,
                          xaiData: _rawXaiData(),
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

  String _formatDateLabel(String? dateStr) {
    if (dateStr == null) return 'Latest Forecast';
    final parsed = DateTime.tryParse(dateStr);
    if (parsed == null) return dateStr;
    final today = DateTime.now();
    final tomorrow = DateTime(today.year, today.month, today.day + 1);
    if (parsed.year == tomorrow.year && parsed.month == tomorrow.month && parsed.day == tomorrow.day) {
      return 'Tomorrow · ${DateFormat('MMM d').format(parsed)}';
    }
    return DateFormat('MMM d, yyyy').format(parsed);
  }

  Future<void> _openDatePicker() async {
    if (availableDates.isEmpty) return;

    final availableDaySet = availableDates
        .map((d) => DateTime.tryParse(d))
        .whereType<DateTime>()
        .map((d) => DateTime(d.year, d.month, d.day))
        .toSet();

    final earliest = availableDaySet.reduce((a, b) => a.isBefore(b) ? a : b);
    final latest = availableDaySet.reduce((a, b) => a.isAfter(b) ? a : b);

    final initial = selectedDate != null
        ? (DateTime.tryParse(selectedDate!) ?? latest)
        : latest;

    final picked = await showDatePicker(
      context: context,
      initialDate: initial,
      firstDate: earliest,
      lastDate: latest,
      selectableDayPredicate: (day) =>
          availableDaySet.contains(DateTime(day.year, day.month, day.day)),
      builder: (context, child) => Theme(
        data: Theme.of(context).copyWith(
          colorScheme: ColorScheme.light(
            primary: AppColors.primaryText,
            onPrimary: Colors.white,
            surface: Colors.white,
            onSurface: Colors.black87,
          ),
        ),
        child: child!,
      ),
    );

    if (picked != null) {
      final pickedStr = DateFormat('yyyy-MM-dd').format(picked);
      if (pickedStr != selectedDate) {
        setState(() => selectedDate = pickedStr);
        _loadData();
      }
    }
  }

  Widget _buildDateButton() {
    return GestureDetector(
      onTap: _openDatePicker,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.grey[50],
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.grey[200]!),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.calendar_month_rounded, size: 18, color: AppColors.primaryText),
            const SizedBox(width: 8),
            Text(
              _formatDateLabel(selectedDate),
              style: GoogleFonts.dmSans(
                fontWeight: FontWeight.w600,
                fontSize: 14,
                color: AppColors.primaryText,
              ),
            ),
            const SizedBox(width: 6),
            Icon(Icons.keyboard_arrow_down_rounded, size: 18, color: Colors.black45),
          ],
        ),
      ),
    );
  }
}
