import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import '../../risk_scoring/utils/constants.dart';
import '../utils/forecast_utils.dart';

class ForecastChart extends StatelessWidget {
  final List<dynamic> values;
  final String pollutant;
  final int selectedHourIndex;
  final Function(int) onHourChanged;
  final String? forecastDate; // Fix 1: actual forecast date for correct time labels

  const ForecastChart({
    super.key,
    required this.values,
    required this.pollutant,
    required this.selectedHourIndex,
    required this.onHourChanged,
    this.forecastDate,
  });

  // Fix 1: use actual forecastDate instead of hardcoding tomorrow
  DateTime _getForecastStart() {
    if (forecastDate != null && forecastDate != 'unknown') {
      return DateTime.tryParse(forecastDate!) ??
          DateTime(DateTime.now().year, DateTime.now().month, DateTime.now().day + 1);
    }
    final now = DateTime.now();
    return DateTime(now.year, now.month, now.day + 1);
  }

  @override
  Widget build(BuildContext context) {
    final forecastStart = _getForecastStart();

    List<FlSpot> spots = [];
    for (int i = 0; i < values.length; i++) {
      spots.add(FlSpot(i.toDouble(), values[i].toDouble()));
    }

    // Fix 2: use correct unit per pollutant
    final unit = ForecastUtils.getUnit(pollutant);

    LineChartBarData lineBarData = LineChartBarData(
      spots: spots,
      isCurved: true,
      preventCurveOverShooting: true,
      color: AppColors.primaryBlue,
      barWidth: 3,
      dotData: FlDotData(show: false),
      belowBarData: BarAreaData(
        show: true,
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [AppColors.primaryBlue.withValues(alpha: 0.2), Colors.transparent],
        ),
      ),
    );

    return Container(
      height: 300,
      margin: const EdgeInsets.symmetric(horizontal: 20),
      padding: const EdgeInsets.fromLTRB(10, 20, 20, 10),
      decoration: BoxDecoration(
        color: AppColors.cardGray,
        borderRadius: BorderRadius.circular(30),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Padding(
                padding: EdgeInsets.only(left: 10, bottom: 10),
                child: Text(
                  "24h Trend",
                  style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                      color: AppColors.primaryText),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                    color: AppColors.primaryBlue,
                    borderRadius: BorderRadius.circular(12)),
                // Fix 2: correct unit in badge
                child: Text(
                  "${values[selectedHourIndex].toStringAsFixed(1)} $unit",
                  style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 12),
                ),
              ),
            ],
          ),

          Expanded(
            child: LineChart(
              LineChartData(
                minY: 0,
                gridData: const FlGridData(show: false),
                titlesData: FlTitlesData(
                  leftTitles:
                      const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  topTitles:
                      const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  rightTitles:
                      const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 36,
                      interval: 6,
                      getTitlesWidget: (val, meta) {
                        final hour = val.toInt();
                        if (hour >= 24) return const SizedBox();
                        // Fix 1: labels use actual forecast date
                        final time =
                            forecastStart.add(Duration(hours: hour));
                        return Padding(
                          padding: const EdgeInsets.only(top: 8.0),
                          child: Text(
                            DateFormat('ha').format(time),
                            style: GoogleFonts.poppins(
                              color: AppColors.primaryText
                                  .withValues(alpha: 0.6),
                              fontSize: 11,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ),
                borderData: FlBorderData(show: false),
                showingTooltipIndicators: [
                  ShowingTooltipIndicators([
                    LineBarSpot(
                        lineBarData, 0, lineBarData.spots[selectedHourIndex]),
                  ]),
                ],
                lineTouchData: LineTouchData(
                  enabled: false,
                  getTouchedSpotIndicator: (barData, spotIndexes) {
                    return spotIndexes.map((index) {
                      return TouchedSpotIndicatorData(
                        const FlLine(
                            color: AppColors.primaryBlue,
                            strokeWidth: 2,
                            dashArray: [4, 4]),
                        FlDotData(
                          show: true,
                          getDotPainter: (spot, percent, bar, index) =>
                              FlDotCirclePainter(
                                  radius: 6,
                                  color: AppColors.primaryBlue,
                                  strokeWidth: 2,
                                  strokeColor: Colors.white),
                        ),
                      );
                    }).toList();
                  },
                  touchTooltipData: LineTouchTooltipData(
                    tooltipBgColor: AppColors.primaryText,
                    getTooltipItems: (List<LineBarSpot> touchedBarSpots) {
                      return touchedBarSpots.map((barSpot) {
                        // Fix 1: tooltip also uses actual forecast date
                        final time = forecastStart
                            .add(Duration(hours: barSpot.x.toInt()));
                        return LineTooltipItem(
                          DateFormat('ha').format(time),
                          const TextStyle(
                              color: Colors.white, fontWeight: FontWeight.bold),
                        );
                      }).toList();
                    },
                  ),
                ),
                lineBarsData: [lineBarData],
              ),
            ),
          ),

          const SizedBox(height: 8),

          // Fix 11: removed "Slide to see specific hour" hint — slider is self-explanatory
          SizedBox(
            height: 30,
            child: SliderTheme(
              data: SliderTheme.of(context).copyWith(
                trackHeight: 2,
                thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
                overlayShape:
                    const RoundSliderOverlayShape(overlayRadius: 14),
              ),
              child: Slider(
                value: selectedHourIndex.toDouble(),
                min: 0,
                max: (values.length - 1).toDouble(),
                divisions: values.length > 1 ? values.length - 1 : 1,
                activeColor: AppColors.primaryBlue,
                inactiveColor:
                    AppColors.lightBlue.withValues(alpha: 0.3),
                onChanged: (val) => onHourChanged(val.toInt()),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
