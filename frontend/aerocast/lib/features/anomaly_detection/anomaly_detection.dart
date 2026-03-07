import 'package:flutter/material.dart';
import 'dart:math';
import 'package:provider/provider.dart';
import 'services/api_service.dart';
import 'models/air_quality.dart';
import 'package:fl_chart/fl_chart.dart';
import 'screens/past_anomalies_screen.dart';

import '../risk_scoring/utils/constants.dart';

class AnomalyScreen extends StatelessWidget {
  const AnomalyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const DashboardMobile();
  }
}

class DashboardMobile extends StatefulWidget {
  const DashboardMobile({super.key});

  @override
  State<DashboardMobile> createState() => _DashboardMobileState();
}

class _DashboardMobileState extends State<DashboardMobile> {
  String selectedLocation = "BATTARAMULLA";
  String selectedPollutant = "O3";
  final ApiService _apiService = ApiService();
  AirQualityData? _currentData;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _initApp();
  }

  Future<void> _initApp() async {
    await _loadData();
  }


  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    try {
      final data = await _apiService.fetchForecast(selectedLocation);
      setState(() {
        _currentData = data;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      if(mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading data: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: _isLoading 
          ? const Center(child: CircularProgressIndicator(color: AppColors.primaryBlue))
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildHeader(),
                  const SizedBox(height: 20),
                  _buildLocationSelector(),
                  const SizedBox(height: 20),
                  _buildPollutantSelector(),
                  const SizedBox(height: 20),
                  _buildMainMetric(),
                  const SizedBox(height: 20),
                  _buildInfoBox(),
                  const SizedBox(height: 20),
                  _buildActionButtons(),
                  const SizedBox(height: 20),
                  _buildWeightedForecast(),
                ],
              ),
            ),
      ),
    );
  }

  Widget _buildHeader() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("Air Quality", style: Theme.of(context).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.bold, color: AppColors.primaryText)),
              const Text(
                "Predicting future air quality spikes",
                style: TextStyle(color: AppColors.secondaryText, fontSize: 13, letterSpacing: 0.2),
              ),
            ],
          ),
        ),
        Row(
          children: [
            IconButton(
              icon: const Icon(Icons.refresh, color: AppColors.primaryBlue),
              onPressed: _initApp,
              style: IconButton.styleFrom(backgroundColor: AppColors.cardGray, elevation: 2),
            ),
            IconButton(
              icon: const Icon(Icons.history_rounded, color: AppColors.primaryBlue),
              onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (c) => const PastAnomaliesScreen())),
              style: IconButton.styleFrom(backgroundColor: AppColors.cardGray, elevation: 2),
            ),
          ],
        )
      ],
    );
  }

  Widget _buildLocationSelector() {
    return Row(
      children: [
        _buildLocationButton("BATTARAMULLA"),
        const SizedBox(width: 10),
        _buildLocationButton("KANDY"),
      ],
    );
  }

  Widget _buildLocationButton(String loc) {
    bool isSelected = selectedLocation == loc;
    return Expanded(
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          backgroundColor: isSelected ? AppColors.primaryBlue : AppColors.cardGray,
          foregroundColor: isSelected ? Colors.white : AppColors.primaryText,
          elevation: isSelected ? 4 : 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: BorderSide(color: isSelected ? AppColors.primaryBlue : Colors.grey.shade300),
          ),
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 8),
        ),
        onPressed: () {
          setState(() => selectedLocation = loc);
          _loadData();
        },
        child: Text(
          loc,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
        ),
      ),
    );
  }

  Widget _buildPollutantSelector() {
    return Container(
      height: 50,
      child: ListView(
        scrollDirection: Axis.horizontal,
        children: ["PM2.5", "PM10", "O3", "SO2"].map((p) {
          bool isSelected = selectedPollutant == p;
          return Padding(
            padding: const EdgeInsets.only(right: 8.0),
            child: InkWell(
              onTap: () => setState(() => selectedPollutant = p),
              borderRadius: BorderRadius.circular(12),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                decoration: BoxDecoration(
                  color: isSelected ? AppColors.darkBlue : AppColors.cardGray,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: isSelected ? AppColors.darkBlue : Colors.grey.shade300),
                  boxShadow: isSelected ? [AppStyles.softShadow] : [],
                ),
                child: Center(
                  child: Text(
                    p,
                    style: TextStyle(
                      color: isSelected ? Colors.white : AppColors.secondaryText,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildMainMetric() {
    double value = _currentData?.getCurrentValue(selectedPollutant) ?? 0.79;
    double threshold = _currentData?.thresholds[selectedPollutant] ?? 1.5;
    
    bool isAnomalous = value > threshold;
    Color boxColor = isAnomalous ? Colors.red.shade400 : Colors.green.shade500;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 40),
      decoration: BoxDecoration(
        color: boxColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [AppStyles.coloredShadow(boxColor)],
      ),
      child: Column(
        children: [
          Text(
            "${value.toStringAsFixed(2)} µg/m³",
            style: const TextStyle(
              color: Colors.white,
              fontSize: 48,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoBox() {
    double value = _currentData?.getCurrentValue(selectedPollutant) ?? 0.79;
    double threshold = _currentData?.thresholds[selectedPollutant] ?? 1.5;
    bool isAnomalous = value > threshold;
    
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isAnomalous ? Colors.red.shade50 : Colors.green.shade50,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: isAnomalous ? Colors.red.shade100 : Colors.green.shade100),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                isAnomalous ? Icons.warning_amber_rounded : Icons.check_circle_outline,
                color: isAnomalous ? Colors.red.shade800 : Colors.green.shade800,
              ),
              const SizedBox(width: 8),
              Text(
                isAnomalous ? "Why is this happening?" : "Ideal Conditions",
                style: TextStyle(
                  color: isAnomalous ? Colors.red.shade800 : Colors.green.shade800,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (isAnomalous) ...[
            ...?_currentData?.reasons[selectedPollutant]?.map((reason) => Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Row(
                children: [
                  Container(width: 8, height: 8, decoration: const BoxDecoration(color: Colors.red, shape: BoxShape.circle)),
                  const SizedBox(width: 12),
                  Expanded(child: Text(reason, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14))),
                ],
              ),
            )),
          ] else
            Text(
              "Pollutant level is stable. No significant risk factors are detected.",
              style: TextStyle(color: Colors.green.shade700),
            ),
        ],
      ),
    );
  }

  Widget _buildActionButtons() {
    return Row(
      children: [
        _buildActionButton(Icons.grid_view_rounded, "Dashboard", AppColors.primaryBlue, () {}),
        const SizedBox(width: 16),
        _buildActionButton(Icons.security_rounded, "Safety Tips", Colors.green.shade500, _showSafetyTips),
      ],
    );
  }

  Widget _buildActionButton(IconData icon, String label, Color color, VoidCallback onTap) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 16),
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [AppStyles.coloredShadow(color)],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, color: Colors.white),
              const SizedBox(width: 8),
              Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ],
          ),
        ),
      ),
    );
  }

  void _showSafetyTips() {
    final tips = _currentData?.safetyTips[selectedPollutant] ?? 
        ["Monitor air quality regularly via apps", "Use air purifiers indoors", "Limit high intensity activities"];
        
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            const Icon(Icons.security, color: Colors.green),
            const SizedBox(width: 10),
            Text("$selectedPollutant Safety Tips"),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: tips
              .map((tip) => ListTile(
                    leading: const CircleAvatar(radius: 12, child: Icon(Icons.check, size: 16)),
                    title: Text(tip),
                  ))
              .toList(),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("Got it!"),
          )
        ],
      ),
    );
  }

  Widget _buildWeightedForecast() {
    final now = DateTime.now();
    final List<String> days = List.generate(7, (index) {
      final date = now.add(Duration(days: index));
      // Simple day name format
      return ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][date.weekday % 7];
    });
    
    final List<double> values = _currentData?.pollutantForecasts[selectedPollutant] ?? [2.7, 1.3, 2.8, 0.5, 1.1, 1.2, 2.7];
    final List<FlSpot> spots = values.asMap().entries.map((e) => FlSpot(e.key.toDouble(), e.value)).toList();
    final double threshold = _currentData?.thresholds[selectedPollutant] ?? 1.5;
    
    // Scaling logic to make threshold "higher" and points more spread
    double maxVal = values.isNotEmpty ? values.reduce((a, b) => a > b ? a : b) : threshold;
    double calculatedMaxY = max(maxVal, threshold) * 1.5; // Ensure threshold is around 66% of height
    if (calculatedMaxY < 1.0) calculatedMaxY = 1.0; 

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.cardGray,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          AppStyles.softShadow,
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("$selectedPollutant Weighted Forecast", style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.primaryText)),
          const SizedBox(height: 20),
          SizedBox(
            height: 220,
            child: ClipRect(
              child: LineChart(
                LineChartData(
                  gridData: FlGridData(
                    show: true,
                    drawVerticalLine: false,
                    getDrawingHorizontalLine: (value) => FlLine(color: Colors.grey.withOpacity(0.1), strokeWidth: 1),
                  ),
                  titlesData: FlTitlesData(
                    show: true,
                    rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 32,
                        interval: 1,
                        getTitlesWidget: (value, meta) {
                          int index = value.toInt();
                          if (index >= 0 && index < days.length) {
                            // Only show labels for exact integers to avoid overlap
                            if (value % 1 != 0) return const Text("");
                            
                            return SideTitleWidget(
                              axisSide: meta.axisSide,
                              space: 10,
                              child: Text(
                                days[index], 
                                style: const TextStyle(
                                  color: AppColors.secondaryText, 
                                  fontSize: 10, // Slightly smaller to ensure fit
                                  fontWeight: FontWeight.w500
                                )
                              ),
                            );
                          }
                          return const Text("");
                        },
                      ),
                    ),
                    leftTitles: const AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: false,
                      ),
                    ),
                  ),
                  lineTouchData: LineTouchData(
                    enabled: true,
                    handleBuiltInTouches: true,
                    touchTooltipData: LineTouchTooltipData(
                      tooltipBgColor: AppColors.primaryText, // Deeper dark for premium look
                      tooltipPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                      tooltipRoundedRadius: 10,
                      getTooltipItems: (List<LineBarSpot> touchedBarSpots) {
                        return touchedBarSpots.map((barSpot) {
                          final index = barSpot.x.toInt();
                          final isAnomaly = barSpot.y > threshold;
                          return LineTooltipItem(
                            "", 
                            const TextStyle(),
                            children: [
                              TextSpan(
                                text: "${days[index]}\n",
                                style: const TextStyle(
                                  color: Colors.white, 
                                  fontWeight: FontWeight.bold, 
                                  fontSize: 15,
                                  height: 1.5
                                ),
                              ),
                              TextSpan(
                                text: "\u25A0 ",
                                style: TextStyle(
                                  color: isAnomaly ? const Color(0xFFFF4D4D) : AppColors.primaryBlue, 
                                  fontSize: 18
                                ),
                              ),
                              TextSpan(
                                text: "Concentration: ${barSpot.y.toStringAsFixed(2)} \u03BCg/m\u00B3",
                                style: const TextStyle(
                                  color: Colors.white, 
                                  fontSize: 13, 
                                  fontWeight: FontWeight.w400
                                ),
                              ),
                            ],
                          );
                        }).toList();
                      },
                    ),
                  ),
                  borderData: FlBorderData(show: false),
                  minX: -0.7, // Room for first label
                  maxX: 6.7,  // Room for last label
                  minY: 0,
                  maxY: calculatedMaxY,
                  extraLinesData: ExtraLinesData(
                    horizontalLines: [
                      HorizontalLine(
                        y: threshold, 
                        color: Colors.red.withOpacity(0.4), 
                        strokeWidth: 1.5, 
                        dashArray: [5, 5],
                      )
                    ],
                  ),
                  lineBarsData: [
                    LineChartBarData(
                      spots: spots,
                      isCurved: true,
                      curveSmoothness: 0.35,
                      color: AppColors.primaryBlue,
                      barWidth: 3,
                      isStrokeCapRound: true,
                      dotData: FlDotData(
                        show: true,
                        getDotPainter: (spot, percent, barData, index) => FlDotCirclePainter(
                          radius: 6,
                          color: spot.y > threshold ? const Color(0xFFFF5252) : AppColors.primaryBlue,
                          strokeWidth: 3,
                          strokeColor: Colors.white,
                        ),
                      ),
                      belowBarData: BarAreaData(
                        show: true,
                        gradient: LinearGradient(
                          colors: [
                            AppColors.primaryBlue.withOpacity(0.2),
                            AppColors.primaryBlue.withOpacity(0.0),
                          ],
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
