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
  String? selectedDate;
  String? forecastDate;

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
      if (!mounted) return;
      setState(() {
        forecastData = result.forecast;
        forecastDate = result.forecastDate;
        selectedHourIndex = 0;
        if (forecastData != null && !forecastData!.containsKey(selectedPollutant)) {
          selectedPollutant = forecastData!.keys.firstWhere((k) => !k.endsWith('_xai'));
        }
      });
    } catch (e) {
      if (forecastData == null) {
        setState(() => isError = true);
      } else if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Couldn't refresh forecast. Showing last loaded data.")),
        );
      }
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
    return DateFormat('EEEE, h:00 a').format(base.add(Duration(hours: selectedHourIndex)));
  }

  Map<String, dynamic>? _rawXaiData() {
    if (forecastData == null) return null;
    final raw = forecastData!["${selectedPollutant}_xai"];
    if (raw == null) return null;
    return Map<String, dynamic>.from(raw);
  }

  String? _topXaiDriver() {
    final raw = _rawXaiData();
    if (raw == null || raw.isEmpty) return null;
    return ForecastUtils.getTopXaiDriver(raw);
  }

// Fix 7: sentence-case location names
  String _formatLocation(String loc) =>
      loc.isEmpty ? loc : loc[0].toUpperCase() + loc.substring(1);

  @override
  Widget build(BuildContext context) {
    // Fix 5: only show full spinner on the very first load (no data yet)
    final bool firstLoad = forecastData == null && !isError;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text(
          "Forecast",
          style: GoogleFonts.dmSans(fontWeight: FontWeight.w800, color: AppColors.primaryText),
        ),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: AppColors.primaryText),
        // Fix 5: thin progress bar while refreshing existing data
        bottom: (isLoading && forecastData != null)
            ? const PreferredSize(
                preferredSize: Size.fromHeight(3),
                child: LinearProgressIndicator(),
              )
            : null,
      ),
      body: firstLoad && isLoading
          ? const Center(child: CircularProgressIndicator())
          : isError && forecastData == null
              ? _buildErrorState()
              // Fix 10: pull-to-refresh
              : RefreshIndicator(
                  onRefresh: _loadData,
                  color: AppColors.primaryBlue,
                  child: SingleChildScrollView(
                    physics: const AlwaysScrollableScrollPhysics(),
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Fix 8: location + date side-by-side
                        Row(
                          children: [
                            Expanded(flex: 3, child: _buildLocationDropdown()),
                            const SizedBox(width: 10),
                            Expanded(flex: 2, child: _buildDateButton()),
                          ],
                        ),
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

                        // Fix 4: removed duplicate "24-Hour Trend" label — chart header already has it

                        if (forecastData != null)
                          ForecastChart(
                            values: forecastData![selectedPollutant],
                            pollutant: selectedPollutant,
                            selectedHourIndex: selectedHourIndex,
                            forecastDate: forecastDate,
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
                ),
    );
  }

  // Fix 6: proper error state with icon + message + retry
  Widget _buildErrorState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.cloud_off_rounded, size: 72, color: Colors.grey[300]),
            const SizedBox(height: 20),
            const Text(
              "Couldn't load forecast",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              "Check your connection and try again",
              style: TextStyle(color: Colors.grey[500], fontSize: 14),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 28),
            ElevatedButton.icon(
              onPressed: _loadData,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text("Retry"),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primaryText,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
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
              // Fix 7: sentence-case instead of all-caps
              child: Text(
                _formatLocation(value),
                style: GoogleFonts.dmSans(
                    fontWeight: FontWeight.w700, fontSize: 14, color: Colors.black87),
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
    if (dateStr == null) return 'Latest';
    final parsed = DateTime.tryParse(dateStr);
    if (parsed == null) return dateStr;
    final today = DateTime.now();
    final tomorrow = DateTime(today.year, today.month, today.day + 1);
    if (parsed.year == tomorrow.year &&
        parsed.month == tomorrow.month &&
        parsed.day == tomorrow.day) {
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

  // Fix 8: date button now fills Expanded space — removed mainAxisSize.min
  Widget _buildDateButton() {
    return GestureDetector(
      onTap: _openDatePicker,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.grey[50],
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.grey[200]!),
        ),
        child: Row(
          children: [
            Icon(Icons.calendar_month_rounded, size: 16, color: AppColors.primaryText),
            const SizedBox(width: 6),
            Expanded(
              child: Text(
                _formatDateLabel(selectedDate),
                style: GoogleFonts.dmSans(
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                  color: AppColors.primaryText,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Icon(Icons.keyboard_arrow_down_rounded, size: 16, color: Colors.black45),
          ],
        ),
      ),
    );
  }
}
